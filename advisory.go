package main

import (
	"bytes"
	"context"
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"io"
	"regexp"
	"runtime"
	"strconv"
	"sync"

	htmltomarkdown "github.com/JohannesKaufmann/html-to-markdown/v2"
	"github.com/ollama/ollama/api"
	"github.com/philippgille/chromem-go"
	"github.com/taylorskalyo/goreader/epub"
	"github.com/tmc/langchaingo/textsplitter"
)

func readItem(w io.Writer, itemRef epub.Itemref) error {
	r, err := itemRef.Open()
	if err != nil {
		return err
	}
	defer r.Close()
	if _, err := io.Copy(w, r); err != nil {
		return err
	}
	return nil
}

func chunkMarkdown(md []byte, metadata map[string]string) ([]Document, error) {
	chunks, err := textsplitter.NewMarkdownTextSplitter().SplitText(string(md))
	if err != nil {
		return nil, err
	}
	if metadata == nil {
		metadata = make(map[string]string)
	}
	documents := make([]Document, len(chunks))
	for i, chunk := range chunks {
		metadata["chunk"] = strconv.Itoa(i)
		documents[i] = Document{
			Metadata: metadata,
			Content:  chunk,
		}
	}
	return documents, nil
}

func ParseEpub(path string) ([]Document, error) {
	rc, err := epub.OpenReader(path)
	if err != nil {
		return nil, err
	}
	defer rc.Close()
	if len(rc.Rootfiles) != 1 {
		return nil, fmt.Errorf("expect 1 root file, got %d", len(rc.Rootfiles))
	}
	book := rc.Rootfiles[0]
	buf := &bytes.Buffer{}
	for _, itemRef := range book.Spine.Itemrefs {
		if err := readItem(buf, itemRef); err != nil {
			return nil, err
		}
	}
	md, err := htmltomarkdown.ConvertReader(buf)
	if err != nil {
		return nil, err
	}
	return chunkMarkdown(md, map[string]string{
		"title":       book.Title,
		"author":      book.Creator,
		"subject":     book.Subject,
		"description": book.Description,
	})
}

type Document struct {
	Content  string
	Metadata map[string]string
}

type Result struct {
	Document
	Score float32
}

type Embedder interface {
	Embed(context.Context, string) ([]float32, error)
}

type ollamaEmbedder struct {
	client *api.Client
	model  string
}

func NewOllamaEmbedder(model string) (Embedder, error) {
	client, err := api.ClientFromEnvironment()
	return ollamaEmbedder{client: client, model: model}, err
}

func (o ollamaEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	res, err := o.client.Embeddings(ctx, &api.EmbeddingRequest{Model: o.model, Prompt: text})
	if err != nil {
		return nil, err
	}
	embeddings := make([]float32, len(res.Embedding))
	for i, e := range res.Embedding {
		embeddings[i] = float32(e)
	}
	return embeddings, nil
}

type Query struct {
	Query  string
	Exact  map[string]string
	Regex  map[string]*regexp.Regexp
	Number int
}

func NewQuery(query string) *Query {
	return (&Query{}).WithQuery(query)
}

func (q *Query) setOrDefault() *Query {
	if q == nil {
		return &Query{Number: 1}
	}
	return q
}

func (q *Query) WithQuery(query string) *Query {
	q = q.setOrDefault()
	q.Query = query
	return q
}

func (q *Query) WithExact(kv ...string) *Query {
	if len(kv)%2 != 0 {
		panic("key-values are not in pairs.")
	}
	exact := make(map[string]string)
	for i := 0; i < len(kv); i += 2 {
		exact[kv[i]] = kv[i+1]
	}
	q = q.setOrDefault()
	q.Exact = exact
	return q
}

func (q *Query) WithRegex(kv ...string) *Query {
	if len(kv)%2 != 0 {
		panic("key-values are not in pairs.")
	}
	regex := make(map[string]*regexp.Regexp)
	for i := 0; i < len(kv); i += 2 {
		regex[kv[i]] = regexp.MustCompile(kv[i+1])
	}
	q = q.setOrDefault()
	q.Regex = regex
	return q
}

func (q *Query) WithNumber(i int) *Query {
	q = q.setOrDefault()
	q.Number = i
	return q
}

func (q *Query) Filter(r Result) bool {
	md := r.Document.Metadata
	switch {
	case len(q.Exact) != 0:
		for k, v := range q.Exact {
			if md[k] != v {
				return false
			}
		}
	case len(q.Regex) != 0:
		for k, r := range q.Regex {
			if !r.MatchString(md[k]) {
				return false
			}
		}
	}
	return true
}

type VectorStore interface {
	Add(context.Context, ...Document) error
	Search(context.Context, *Query) ([]Result, error)
}

type chromemStore struct {
	collection  *chromem.Collection
	concurrency int
}

func NewCollection(path, collection string, e Embedder) (VectorStore, error) {
	db, err := chromem.NewPersistentDB(path, false)
	if err != nil {
		return nil, err
	}
	c, err := db.GetOrCreateCollection(collection, make(map[string]string), e.Embed)
	if err != nil {
		return nil, err
	}
	return &chromemStore{collection: c, concurrency: runtime.NumCPU()}, nil
}

func md5hash(s string) string {
	hash := md5.Sum([]byte(s))
	return hex.EncodeToString(hash[:])
}

func (c *chromemStore) Add(ctx context.Context, docs ...Document) error {
	documents := make([]chromem.Document, len(docs))
	sem := make(chan struct{}, c.concurrency)
	wg := &sync.WaitGroup{}
	for i, doc := range docs {
		sem <- struct{}{}
		wg.Add(1)
		go func() {
			documents[i] = chromem.Document{
				ID:       md5hash(doc.Content),
				Metadata: doc.Metadata,
				Content:  doc.Content,
			}
			<-sem
			wg.Done()
		}()
	}
	wg.Wait()
	return c.collection.AddDocuments(ctx, documents, c.concurrency)
}

func (c *chromemStore) Search(ctx context.Context, query *Query) ([]Result, error) {
	q := query.setOrDefault()
	opt := chromem.QueryOptions{
		QueryText: q.Query,
		Where:     q.Exact,
		NResults:  q.Number,
	}
	res, err := c.collection.QueryWithOptions(ctx, opt)
	if err != nil {
		return nil, err
	}
	results := make([]Result, 0, len(res))
	for _, r := range res {
		result := Result{
			Document: Document{
				Metadata: r.Metadata,
				Content:  r.Content,
			},
			Score: r.Similarity,
		}
		if query.Filter(result) {
			results = append(results, result)
		}
	}
	return results, nil
}
