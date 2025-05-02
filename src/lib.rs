pub mod embeddings;
pub use crate::embeddings::Embeddings;

use kitoken::Kitoken;
use std::{boxed::Box, mem, slice, str};

/// Creates a new Kitoken instance from serialized definition bytes.
///
/// # Safety
/// `data_ptr` must be a valid pointer to `data_len` bytes of a serialized Kitoken definition.
/// The memory pointed to by `data_ptr` must remain valid for the duration of this call.
///
/// # Returns
/// A raw pointer to the created Kitoken instance, or null if creation fails.
/// The caller is responsible for freeing this instance using `kitoken_free`.
#[no_mangle]
pub extern "C" fn kitoken_new(data_ptr: *const u8, data_len: usize) -> *mut Kitoken {
    // SAFETY: Assumes `data_ptr` is valid for `data_len` bytes as per function contract.
    let data = unsafe { slice::from_raw_parts(data_ptr, data_len) };
    match Kitoken::from_slice(data) {
        // Box the instance and leak the pointer to the C side.
        Ok(t) => Box::into_raw(Box::new(t)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Frees a Kitoken instance previously created by `kitoken_new`.
///
/// # Safety
/// `ptr` must be a valid pointer obtained from `kitoken_new` or null.
/// Calling this function with an invalid pointer or a pointer that has already been freed
/// results in undefined behavior.
#[no_mangle]
pub extern "C" fn kitoken_free(ptr: *mut Kitoken) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: Assumes `ptr` is a valid pointer obtained from `kitoken_new`
    //         and has not been freed yet, as per function contract.
    //         Retakes ownership of the Box and drops it, freeing the memory.
    unsafe {
        drop(Box::from_raw(ptr));
    }
}

/// Encodes a UTF-8 string into token IDs using the provided Kitoken instance.
///
/// # Safety
/// - `ptr` must be a valid pointer to a Kitoken instance obtained from `kitoken_new`.
/// - `text_ptr` must be a valid pointer to `text_len` bytes of UTF-8 encoded text.
/// - The memory pointed to by `ptr` and `text_ptr` must remain valid for the duration of this call.
/// - `out_tokens_ptr` and `out_len_ptr` must be valid pointers to `*mut u32` and `usize` respectively.
///
/// # Returns
/// Returns 0 on success, -1 on error (e.g., invalid UTF-8).
/// On success, `out_tokens_ptr` will point to the allocated array of token IDs (u32),
/// and `out_len_ptr` will contain the number of tokens.
/// The caller is responsible for freeing the memory pointed to by `out_tokens_ptr`
/// using `dealloc_u32` with the pointer and the length.
#[no_mangle]
pub extern "C" fn kitoken_encode(
    ptr: *mut Kitoken,
    text_ptr: *const u8,
    text_len: usize,
    out_tokens_ptr: *mut *mut u32,
    out_len_ptr: *mut usize,
) -> i32 {
    if ptr.is_null() || text_ptr.is_null() || out_tokens_ptr.is_null() || out_len_ptr.is_null() {
        return -1; // Basic null pointer check
    }
    // SAFETY: Assumes `ptr` is a valid Kitoken pointer and `text_ptr` is valid for `text_len` bytes.
    let tokenizer = unsafe { &*ptr };
    let bytes = unsafe { slice::from_raw_parts(text_ptr, text_len) };

    let text = match str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => return -1, // Invalid UTF-8
    };

    match tokenizer.encode(text, false) {
        Ok(tokens) => {
            let mut boxed = tokens.into_boxed_slice();
            let len = boxed.len();
            let ptr_out = boxed.as_mut_ptr();
            // Transfer ownership of the allocated memory to the C caller.
            mem::forget(boxed);

            // SAFETY: Assumes `out_tokens_ptr` and `out_len_ptr` are valid pointers.
            unsafe {
                *out_len_ptr = len;
                *out_tokens_ptr = ptr_out;
            }
            0 // Success
        }
        Err(_) => -1, // Encoding error
    }
}

/// Decodes token IDs back into a UTF-8 byte vector using the provided Kitoken instance.
///
/// # Safety
/// - `ptr` must be a valid pointer to a Kitoken instance obtained from `kitoken_new`.
/// - `tokens_ptr` must be a valid pointer to `tokens_len` u32 token IDs.
/// - The memory pointed to by `ptr` and `tokens_ptr` must remain valid for the duration of this call.
/// - `out_ptr` and `out_len_ptr` must be valid pointers to `*mut u8` and `usize` respectively.
///
/// # Returns
/// Returns 0 on success, -1 on error.
/// On success, `out_ptr` will point to the allocated byte array (UTF-8 encoded),
/// and `out_len_ptr` will contain the number of bytes.
/// The caller is responsible for freeing the memory pointed to by `out_ptr`
/// using `dealloc` with the pointer and the length.
#[no_mangle]
pub extern "C" fn kitoken_decode(
    ptr: *mut Kitoken,
    tokens_ptr: *const u32,
    tokens_len: usize,
    out_ptr: *mut *mut u8,
    out_len_ptr: *mut usize,
) -> i32 {
    if ptr.is_null() || tokens_ptr.is_null() || out_ptr.is_null() || out_len_ptr.is_null() {
        return -1; // Basic null pointer check
    }
    // SAFETY: Assumes `ptr` is valid and `tokens_ptr` is valid for `tokens_len` u32s.
    let tokenizer = unsafe { &*ptr };
    let tokens = unsafe { slice::from_raw_parts(tokens_ptr, tokens_len) };

    match tokenizer.decode(tokens, false) {
        Ok(vec) => {
            let mut boxed = vec.into_boxed_slice();
            let len = boxed.len();
            let ptr_out = boxed.as_mut_ptr();
            // Transfer ownership of the allocated memory to the C caller.
            mem::forget(boxed);

            // SAFETY: Assumes `out_ptr` and `out_len_ptr` are valid pointers.
            unsafe {
                *out_len_ptr = len;
                *out_ptr = ptr_out;
            }
            0 // Success
        }
        Err(_) => -1, // Decoding error
    }
}

/// Creates a new Embeddings instance from serialized model data.
///
/// # Safety
/// `model_data_ptr` must be a valid pointer to `model_data_len` bytes of a serialized rten model.
/// The memory pointed to by `model_data_ptr` must remain valid for the duration of this call.
///
/// # Returns
/// A raw pointer to the created Embeddings instance, or null if creation fails.
/// The caller is responsible for freeing this instance using `embeddings_free`.
#[no_mangle]
pub extern "C" fn embeddings_new(
    model_data_ptr: *const u8,
    model_data_len: usize,
) -> *mut Embeddings {
    if model_data_ptr.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: Assumes `model_data_ptr` is valid for `model_data_len` bytes.
    let model_data_slice = unsafe { slice::from_raw_parts(model_data_ptr, model_data_len) };
    // Copy the model data into a Vec owned by Rust.
    let model_data_vec = model_data_slice.to_vec();

    match Embeddings::new(model_data_vec) {
        // Box the instance and leak the pointer to the C side.
        Ok(e) => Box::into_raw(Box::new(e)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Frees an Embeddings instance previously created by `embeddings_new`.
///
/// # Safety
/// `ptr` must be a valid pointer obtained from `embeddings_new` or null.
/// Calling this function with an invalid pointer or a pointer that has already been freed
/// results in undefined behavior.
#[no_mangle]
pub extern "C" fn embeddings_free(ptr: *mut Embeddings) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: Assumes `ptr` is valid and hasn't been freed.
    //         Retakes ownership of the Box and drops it.
    unsafe {
        drop(Box::from_raw(ptr));
    }
}

/// Computes embeddings for a single token ID sequence.
///
/// # Safety
/// - `ptr` must be a valid pointer to an Embeddings instance obtained from `embeddings_new`.
/// - `ids_ptr` must be a valid pointer to an array of `len` i32 token IDs.
/// - The memory pointed to by `ptr` and `ids_ptr` must be valid for the duration of the call.
/// - `out_ptr`, `out_count_ptr`, `out_dim_ptr` must be valid pointers to `*mut f32`, `usize`, `usize`.
///
/// # Returns
/// Returns 0 on success, -1 on error.
/// On success, `out_ptr` points to a flat f32 array containing the embedding vector
/// for the input sequence. `out_count_ptr` will contain 1 (the batch size, which is always 1),
/// and `out_dim_ptr` contains the embedding dimension.
/// The caller is responsible for freeing the memory pointed to by `out_ptr`
/// using `dealloc_f32` with the pointer and the total number of elements (`1 * out_dim_ptr`, which is just `out_dim_ptr`).
#[no_mangle]
pub extern "C" fn embeddings_embed(
    ptr: *mut Embeddings,
    ids_ptr: *const i32,
    len: usize,
    out_ptr: *mut *mut f32,
    out_count_ptr: *mut usize,
    out_dim_ptr: *mut usize,
) -> i32 {
    if ptr.is_null()
        || ids_ptr.is_null()
        || out_ptr.is_null()
        || out_count_ptr.is_null()
        || out_dim_ptr.is_null()
    {
        return -1;
    }
    // SAFETY: ids_ptr is valid for len elements
    let embedder = unsafe { &*ptr };
    let ids_slice = unsafe { slice::from_raw_parts(ids_ptr, len) };
    match embedder.embed(vec![ids_slice.to_vec()]) {
        Ok(result_vecs) => {
            let batch = result_vecs.len();
            let dim = if batch > 0 { result_vecs[0].len() } else { 0 };
            let total = batch * dim;
            // Flatten
            let mut flat_vec: Vec<f32> = Vec::with_capacity(total);
            for v in result_vecs.iter() {
                if v.len() != dim {
                    return -1; // inconsistent dims
                }
                flat_vec.extend_from_slice(v);
            }
            let mut boxed = flat_vec.into_boxed_slice();
            let ptr_out = boxed.as_mut_ptr();
            mem::forget(boxed);
            // Write outputs
            unsafe {
                *out_count_ptr = batch;
                *out_dim_ptr = dim;
                *out_ptr = ptr_out;
            }
            0
        }
        Err(_) => -1,
    }
}

/// Allocates a memory buffer of the specified size for FFI communication (typically for receiving data from Rust).
///
/// The caller receives ownership of the allocated buffer and is responsible for eventually
/// deallocating it using the appropriate deallocation function (`dealloc` for u8, `dealloc_u32` for u32, etc.)
/// with the returned pointer and the original size/length.
///
/// The allocated memory is **uninitialized**.
///
/// # Safety
/// The caller must ensure `size` does not cause allocation failure (e.g., too large).
///
/// # Returns
/// A pointer to the allocated buffer, or potentially null if allocation fails (though `Vec::with_capacity` might panic on extreme sizes).
#[no_mangle]
pub extern "C" fn alloc(size: usize) -> *mut u8 {
    let mut buf = Vec::with_capacity(size);
    let ptr = buf.as_mut_ptr();
    // Leak the Vec's buffer, transferring ownership to the caller.
    mem::forget(buf);
    ptr
}

/// Deallocates a `u8` memory buffer previously allocated by a Rust FFI function (like `kitoken_decode`).
///
/// # Safety
/// - `ptr` must be the pointer returned by the corresponding allocation function.
/// - `size` must be the exact size/length that was associated with the buffer when allocated.
/// - Calling this function with invalid parameters (wrong pointer, wrong size, double free)
///   results in undefined behavior.
#[no_mangle]
pub extern "C" fn dealloc(ptr: *mut u8, size: usize) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: Assumes `ptr` and `size` are correct as per the function contract,
    //         reconstructing the Vec and allowing it to be dropped.
    unsafe {
        Vec::from_raw_parts(ptr, 0, size);
    }
}

/// Deallocates a `u32` memory buffer previously allocated by a Rust FFI function (like `kitoken_encode`).
///
/// # Safety
/// - `ptr` must be the pointer returned by the corresponding allocation function (`kitoken_encode`).
/// - `len` must be the exact number of `u32` elements that was associated with the buffer when allocated.
/// - Calling this function with invalid parameters (wrong pointer, wrong length, double free)
///   results in undefined behavior.
#[no_mangle]
pub extern "C" fn dealloc_u32(ptr: *mut u32, len: usize) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: Assumes `ptr` and `len` are correct, reconstructing the Vec<u32>
    //         from the raw parts and allowing it to be dropped.
    unsafe {
        let element_size = std::mem::size_of::<u32>();
        Vec::from_raw_parts(ptr, 0, len * element_size); // Capacity is in bytes
                                                         // Note: While Vec::from_raw_parts takes capacity in bytes, reconstructing a Vec<T>
                                                         // with the element pointer and element count (len) is the standard, safe way.
                                                         // Let's rewrite using the standard approach for Vec<T>.
        Vec::from_raw_parts(ptr, 0, len);
    }
}

/// Deallocates an `f32` memory buffer previously allocated by a Rust FFI function (like `embeddings_embed`).
///
/// # Safety
/// - `ptr` must be the pointer returned by `embeddings_embed`.
/// - `len` must be the exact total number of `f32` elements (`batch_size * embedding_dim`)
///   that was associated with the buffer when allocated.
/// - Calling this function with invalid parameters (wrong pointer, wrong length, double free)
///   results in undefined behavior.
#[no_mangle]
pub extern "C" fn dealloc_f32(ptr: *mut f32, len: usize) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: Assumes `ptr` and `len` are correct, reconstructing the Vec<f32>
    //         from the raw parts and allowing it to be dropped.
    unsafe {
        Vec::from_raw_parts(ptr, 0, len);
    }
}
