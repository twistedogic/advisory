package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/huh"
	"github.com/charmbracelet/huh/spinner"
	"github.com/google/subcommands"
)

const (
	defaultEmbeddingModel = "nomic-embed-text"
	defaultStoreName      = "default"
)

type chromaStoreCmd struct {
	model, storeName string
}

func (c *chromaStoreCmd) SetFlags(f *flag.FlagSet) {
	f.StringVar(&c.model, "model", defaultEmbeddingModel, "embedding model to use")
	f.StringVar(&c.storeName, "collection", defaultStoreName, "collection of vectorstore")
}

func (c *chromaStoreCmd) Store() (VectorStore, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}
	storePath := filepath.Join(home, ".advisory", "store")
	if err := os.MkdirAll(storePath, 0755); err != nil {
		return nil, err
	}
	e, err := NewOllamaEmbedder(c.model)
	if err != nil {
		return nil, err
	}

	return NewCollection(storePath, c.model+"_"+c.storeName, e)
}

type addCmd struct{ chromaStoreCmd }

func (*addCmd) Name() string     { return "add" }
func (*addCmd) Synopsis() string { return "add epub file content to vectorstore" }
func (*addCmd) Usage() string    { return "" }

func (a *addCmd) Execute(ctx context.Context, f *flag.FlagSet, _ ...interface{}) subcommands.ExitStatus {
	patterns := f.Args()
	if len(patterns) == 0 {
		slog.Error("no file(s) provided.")
		return subcommands.ExitUsageError
	}
	store, err := a.Store()
	if err != nil {
		slog.Error(err.Error())
		return subcommands.ExitFailure
	}
	if err := spinner.New().
		Context(ctx).
		ActionWithErr(func(ctx context.Context) error {
			for _, pattern := range patterns {
				matches, err := filepath.Glob(pattern)
				if err != nil {
					return err
				}
				slog.Info(fmt.Sprintf("found %d files", len(matches)))
				for _, path := range matches {
					docs, err := ParseEpub(path)
					if err != nil {
						return fmt.Errorf("parse %q: %w", path, err)
					}
					slog.Info(fmt.Sprintf("import %d chunks", len(docs)))
					if err := store.Add(ctx, docs...); err != nil {
						return err
					}
				}
			}
			return nil
		}).
		Accessible(false).
		Run(); err != nil {
		slog.Error(err.Error())
		return subcommands.ExitFailure
	}

	return subcommands.ExitSuccess
}

type queryCmd struct {
	chromaStoreCmd
	nResult int
}

func (*queryCmd) Name() string     { return "query" }
func (*queryCmd) Synopsis() string { return "query vectorstore" }
func (*queryCmd) Usage() string    { return "" }
func (q *queryCmd) SetFlags(f *flag.FlagSet) {
	q.chromaStoreCmd.SetFlags(f)
	f.IntVar(&q.nResult, "n", 10, "number of results")
}

func (q *queryCmd) Execute(ctx context.Context, _ *flag.FlagSet, _ ...interface{}) subcommands.ExitStatus {
	var query string
	form := huh.NewForm(
		huh.NewGroup(huh.NewInput().Title("query").Value(&query)),
	)
	if err := form.Run(); err != nil {
		slog.Error(err.Error())
		return subcommands.ExitFailure
	}
	content := []string{}
	if err := spinner.New().Context(ctx).Accessible(false).ActionWithErr(func(ctx context.Context) error {
		store, err := q.Store()
		if err != nil {
			return err
		}
		results, err := store.Search(ctx, NewQuery(query).WithNumber(q.nResult))
		if err != nil {
			return err
		}
		for _, result := range results {
			content = append(content, fmt.Sprintf(
				"> %s by %s\n\n%s",
				result.Metadata["title"],
				result.Metadata["author"],
				result.Content,
			))
		}
		return nil
	}).Run(); err != nil {
		slog.Error(err.Error())
		return subcommands.ExitFailure
	}
	md, err := glamour.Render(strings.Join(content, "\n--------\n"), "dark")
	if err != nil {
		slog.Error(err.Error())
		return subcommands.ExitFailure
	}
	fmt.Println(md)
	return subcommands.ExitSuccess
}

func main() {
	subcommands.Register(subcommands.HelpCommand(), "")
	subcommands.Register(subcommands.FlagsCommand(), "")
	subcommands.Register(subcommands.CommandsCommand(), "")
	subcommands.Register(&addCmd{}, "")
	subcommands.Register(&queryCmd{}, "")

	flag.Parse()
	ctx := context.Background()
	os.Exit(int(subcommands.Execute(ctx)))
}
