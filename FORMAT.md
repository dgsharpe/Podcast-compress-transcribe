# `.mimi` file format, v1

This is a seekable container for Mimi (kyutai/mimi) RVQ codes. The MIME type is `audio/x-mimi`.

Every frame has the same size, so the byte offset of any time position is computed directly; no index is needed.
That makes seeking work with an HTTP Range request or a plain `pread`.

```
offset  size  field
0       4     magic "MIMI"
4       1     version = 1
5       1     nq            number of codebooks per frame (1..32; 32 = 4.4 kb/s)
6       1     bits          bits per code (11 for Mimi's 2048-entry codebooks)
7       1     flags         0 (reserved)
8       4     sample_rate   24000 (u32 LE)
12      2     fr_num        25    (frame rate = fr_num / fr_den = 12.5 Hz)
14      2     fr_den        2
16      4     n_frames      (u32 LE)
20      4     frame_bytes   ceil(nq * bits / 8) = 44 for nq = 32
24      4     data_offset   byte offset of frame 0 (u32 LE)
28      4     meta_len      length of the UTF-8 JSON metadata that starts at byte 64
32      32    model         NUL-padded ASCII model id, e.g. "kyutai/mimi"
64      ...   metadata JSON (title, source URL, encoder settings, ...), then zero padding up to data_offset
data_offset   n_frames * frame_bytes   frames
```

Frame `f` starts at `data_offset + f * frame_bytes`. Its codes are bit-packed little-endian: code `i` (codebook
`i`; 0 is the semantic codebook) occupies bits `[i*bits, (i+1)*bits)` of the frame, LSB first. Each frame
is 1/12.5 s = 80 ms (1,920 samples at 24 kHz).

`data_offset` is a multiple of 64. The decoder needs no state from earlier frames beyond the left-context
frames it reads itself (see `phone/libmimi`), so playback can start at any frame.
