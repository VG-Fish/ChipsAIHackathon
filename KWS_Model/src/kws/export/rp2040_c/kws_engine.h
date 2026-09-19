/* Integer-only SparkNet inference (see kws/export/rp2040.py for the spec).
 *
 * No floating point, no heap, no libc beyond <stdint.h>/<stddef.h>: the same
 * source builds for the host parity harness and for the RP2040 firmware.
 * Tensors are [channel][frame], int16 storage for 8- and 16-bit activations.
 */
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct {
    int16_t cin, cout, kernel;
    const int8_t *dw_w;     /* [cin][kernel] */
    const int32_t *dw_m;    /* depthwise requant, per input channel */
    const int64_t *dw_b;
    const uint8_t *dw_s;
    const int8_t *pw_w;     /* [cout][cin], on the depthwise output */
    const int8_t *res_w;    /* [cout][cin], on the block input; NULL for block 0 */
    const int8_t *den_w;    /* [cout][cin] dendrite, on the depthwise output; NULL if none */
    const int32_t *den_m;   /* dendrite accumulator -> tanh LUT index */
    const int64_t *den_b;
    const uint8_t *den_s;
    const int16_t *den_lut; /* 255 entries, tanh * 32767 */
    const int32_t *out_m1;  /* output requant: pointwise, residual, tanh terms */
    const int32_t *out_m2;
    const int32_t *out_m3;
    const int64_t *out_b;
    const uint8_t *out_s;
} kws_block_t;

typedef struct {
    int16_t n_feat, n_frames, channels, gate_channels, n_classes, n_blocks;
    int16_t act_max, relu_max;
    const kws_block_t *blocks;
    const int8_t *gate_w;   /* [gate_channels][channels] */
    const int32_t *gate_m;
    const int64_t *gate_b;
    const uint8_t *gate_s;
    const uint8_t *gate_lut; /* 255 entries, clamp(tanh + 0.5, 0, 1) * 255 */
    const int8_t *fc_w;     /* [n_classes][gate_channels] */
    const int32_t *fc_m;    /* logit requant: fc and fc dendrite tanh terms */
    const int64_t *fc_b;
    const uint8_t *fc_s;
    const int32_t *fc_m2;   /* NULL (with every fc_den_*) when fc has no dendrite */
    const int8_t *fc_den_w; /* [n_classes][gate_channels] dendrite, on the gate sums */
    const int32_t *fc_den_m; /* fc dendrite accumulator -> tanh LUT index */
    const int64_t *fc_den_b;
    const uint8_t *fc_den_s;
    const int16_t *fc_den_lut; /* 255 entries, tanh * 32767 */
    const float *input_scale; /* frontend only: x_q = round(mfcc / input_scale) */
} kws_model_t;

/* int16 elements of scratch kws_run needs: three [max(n_feat, channels)][n_frames] buffers. */
size_t kws_scratch_elems(const kws_model_t *m);

/* Run one clip. input: [n_feat][n_frames] integers already quantized with
 * input_scale; logits: n_classes Q16 values.  Returns the argmax class. */
int kws_run(const kws_model_t *m, const int16_t *input, int32_t *logits, int16_t *scratch);
