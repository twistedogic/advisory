package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func Test_chromemStore(t *testing.T) {
	e, err := NewOllamaEmbedder(embeddingModel)
	require.NoError(t, err)
	dir, err := os.MkdirTemp("", "chromem_test")
	require.NoError(t, err)
	storePath := filepath.Join(dir, "store")
	defer os.RemoveAll(dir)
	c, err := NewCollection(storePath, "default", e)
	require.NoError(t, err)
	chunks, err := ParseEpub("testdata/pg35542.epub")
	require.NoError(t, err)
	require.NotEmpty(t, chunks)
	require.NoError(t, c.Add(t.Context(), chunks...))
	documents := []Document{
		{
			Content: "The sky is blue because of Rayleigh scattering.",
		},
		{
			Content: "Leaves are green because chlorophyll absorbs red and blue light.",
		},
	}
	require.NoError(t, c.Add(t.Context(), documents...))
	results, err := c.Search(t.Context(), NewQuery("Why is the sky blue?").WithNumber(1))
	require.NoError(t, err)
	require.Len(t, results, 1)
	require.Equal(t, results[0].Content, documents[0].Content)
	results, err = c.Search(t.Context(), NewQuery("What are the natural enemy of rat?").WithNumber(10))
	require.NoError(t, err)
	require.Len(t, results, 10)
	t.Log(results)
}
