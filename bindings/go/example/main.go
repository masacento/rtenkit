package main

import (
	"fmt"
	"log"

	rtenkit "github.com/masacento/rtenkit/bindings/go"
)

func main() {
	embedder, err := rtenkit.NewEmbedder("./modeldata/model.rten", "./modeldata/tokenizer.kitoken")
	if err != nil {
		log.Fatalf("Error creating embedder: %v\n", err)
	}
	defer embedder.Close()

	emb, err := embedder.Embed("hello world")
	if err != nil {
		log.Fatalf("Error embedding: %v\n", err)
	}

	fmt.Println(emb)
}
