package rtenkit

/*
#include <stdlib.h>
#include <stdint.h>

// Kitoken functions
extern void* kitoken_new(const uint8_t* data_ptr, size_t data_len);
extern void kitoken_free(void* ptr);
extern int32_t kitoken_encode(void* ptr, const uint8_t* text_ptr, size_t text_len, void** out_tokens_ptr, size_t* out_len_ptr);
extern int32_t kitoken_decode(void* ptr, const uint32_t* tokens_ptr, size_t tokens_len, void** out_ptr, size_t* out_len_ptr);

// Embeddings functions
extern void* embeddings_new(const uint8_t* data_ptr, size_t data_len);
extern void embeddings_free(void* ptr);
extern int32_t embeddings_embed(void* ptr, const int32_t* ids_ptr, size_t len, void** out_ptr, size_t* out_count_ptr, size_t* out_dim_ptr);

// Deallocation functions
extern void dealloc(void* ptr, size_t size);
extern void dealloc_f32(void* ptr, size_t len);
*/
import "C"

import (
	_ "embed"
	"fmt"
	"os"
	"runtime"
	"unsafe"
)

type Embedder struct {
	tokenizer  *Kitoken
	embeddings *Embeddings
}

func NewEmbedder(modelpath, tokenizerpath string) (*Embedder, error) {
	modeldata, err := os.ReadFile(modelpath)
	if err != nil {
		return nil, err
	}
	emb, err := NewEmbeddings(modeldata)
	if err != nil {
		return nil, err
	}

	tokenizerdata, err := os.ReadFile(tokenizerpath)
	if err != nil {
		return nil, err
	}

	tk, err := NewKitoken(tokenizerdata)
	if err != nil {
		return nil, err
	}

	return &Embedder{tokenizer: tk, embeddings: emb}, nil
}

func (e *Embedder) Embed(s string) ([]float32, error) {
	tokens, err := e.tokenizer.Encode(s)
	if err != nil {
		return nil, err
	}
	return e.embeddings.EmbedTokens(tokens)
}

func (e *Embedder) Close() {
	e.tokenizer.Free()
	e.embeddings.Free()
}

type Kitoken struct {
	ptr unsafe.Pointer
}

func NewKitoken(data []byte) (*Kitoken, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("empty model data")
	}
	ptr := C.kitoken_new((*C.uint8_t)(unsafe.Pointer(&data[0])), C.size_t(len(data)))
	if ptr == nil {
		return nil, fmt.Errorf("failed to create tokenizer")
	}
	tk := &Kitoken{ptr: ptr}
	runtime.SetFinalizer(tk, func(t *Kitoken) { t.Free() })
	return tk, nil
}

func (t *Kitoken) Free() {
	if t.ptr != nil {
		C.kitoken_free(t.ptr)
		t.ptr = nil
	}
}

func (t *Kitoken) Encode(text string) ([]uint32, error) {
	if t.ptr == nil {
		return nil, fmt.Errorf("tokenizer is closed")
	}
	b := []byte(text)
	var outPtr unsafe.Pointer
	var outLen C.size_t
	res := C.kitoken_encode(t.ptr, (*C.uint8_t)(unsafe.Pointer(&b[0])), C.size_t(len(b)), &outPtr, &outLen)
	if res != 0 {
		return nil, fmt.Errorf("encoding failed")
	}
	length := int(outLen)

	cArr := unsafe.Slice((*C.uint32_t)(outPtr), length)
	tokens := make([]uint32, length)
	for i := 0; i < length; i++ {
		tokens[i] = uint32(cArr[i])
	}

	C.dealloc(outPtr, C.size_t(length)*C.size_t(unsafe.Sizeof(C.uint32_t(0))))
	return tokens, nil
}

func (t *Kitoken) Decode(tokens []uint32) (string, error) {
	if t.ptr == nil {
		return "", fmt.Errorf("tokenizer is closed")
	}
	if len(tokens) == 0 {
		return "", nil
	}
	var outPtr unsafe.Pointer
	var outLen C.size_t
	res := C.kitoken_decode(t.ptr, (*C.uint32_t)(unsafe.Pointer(&tokens[0])), C.size_t(len(tokens)), &outPtr, &outLen)
	if res != 0 {
		return "", fmt.Errorf("decoding failed")
	}
	length := int(outLen)
	bytes := unsafe.Slice((*byte)(outPtr), length)

	result := string(bytes)
	C.dealloc(outPtr, C.size_t(outLen))
	return result, nil
}

type Embeddings struct {
	ptr unsafe.Pointer
}

func NewEmbeddings(data []byte) (*Embeddings, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("empty embeddings model data")
	}
	ptr := C.embeddings_new((*C.uint8_t)(unsafe.Pointer(&data[0])), C.size_t(len(data)))
	if ptr == nil {
		return nil, fmt.Errorf("failed to create embeddings")
	}
	emb := &Embeddings{ptr: ptr}
	runtime.SetFinalizer(emb, func(e *Embeddings) { e.Free() })
	return emb, nil
}

func (e *Embeddings) Free() {
	if e.ptr != nil {
		C.embeddings_free(e.ptr)
		e.ptr = nil
	}
}

func (e *Embeddings) EmbedTokens(tokens []uint32) ([]float32, error) {
	if e.ptr == nil {
		return nil, fmt.Errorf("embeddings is closed")
	}
	if len(tokens) == 0 {
		return nil, nil
	}

	intTokens := make([]C.int32_t, len(tokens))
	for i, t := range tokens {
		intTokens[i] = C.int32_t(t)
	}
	var outPtr unsafe.Pointer
	var outCount C.size_t
	var outDim C.size_t

	res := C.embeddings_embed(e.ptr, (*C.int32_t)(unsafe.Pointer(&intTokens[0])), C.size_t(len(intTokens)), &outPtr, &outCount, &outDim)
	if res != 0 {
		return nil, fmt.Errorf("embedding failed")
	}
	count := int(outCount)
	dim := int(outDim)
	total := count * dim
	cArr := unsafe.Slice((*C.float)(outPtr), total)
	result := make([]float32, total)
	for i := 0; i < total; i++ {
		result[i] = float32(cArr[i])
	}
	C.dealloc_f32(outPtr, C.size_t(total))
	return result, nil
}
