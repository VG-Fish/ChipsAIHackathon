/* Integer-only SparkNet inference; mirrors QuantizedSparkNet.forward_int. */
#include "kws_engine.h"

#define LUT_HALF 127

static inline int32_t clamp32(int64_t v, int32_t lo, int32_t hi) {
    return v < lo ? lo : (v > hi ? hi : (int32_t)v);
}

size_t kws_scratch_elems(const kws_model_t *m) {
    int16_t widest = m->n_feat > m->channels ? m->n_feat : m->channels;
    return (size_t)3 * (size_t)widest * (size_t)m->n_frames;
}

/* 'same' zero-padded depthwise conv + requant to signed activations */
static void depthwise(const kws_block_t *b, const int16_t *x, int16_t *d, int T, int32_t act_max) {
    const int K = b->kernel, pad = K / 2;
    for (int c = 0; c < b->cin; c++) {
        const int8_t *w = b->dw_w + c * K;
        const int16_t *xc = x + c * T;
        const int64_t m = b->dw_m[c], bias = b->dw_b[c];
        const int s = b->dw_s[c];
        for (int t = 0; t < T; t++) {
            int k0 = pad - t > 0 ? pad - t : 0;
            int k1 = T + pad - t < K ? T + pad - t : K;
            int32_t acc = 0;
            for (int k = k0; k < k1; k++) acc += (int32_t)w[k] * xc[t - pad + k];
            d[c * T + t] = (int16_t)clamp32(((int64_t)acc * m + bias) >> s, -act_max, act_max);
        }
    }
}

static inline int32_t dot(const int8_t *w, const int16_t *x, int n, int stride) {
    int32_t acc = 0;
    for (int i = 0; i < n; i++) acc += (int32_t)w[i] * x[i * stride];
    return acc;
}

static void block_out(const kws_block_t *b, const int16_t *x, const int16_t *d, int16_t *y, int T,
                      int32_t relu_max) {
    const int cin = b->cin;
    for (int o = 0; o < b->cout; o++) {
        const int8_t *pw = b->pw_w + o * cin;
        const int8_t *rw = b->res_w ? b->res_w + o * cin : 0;
        const int8_t *dn = b->den_w ? b->den_w + o * cin : 0;
        const int64_t m1 = b->out_m1[o], m2 = b->out_m2[o], m3 = b->out_m3[o], bias = b->out_b[o];
        const int s = b->out_s[o];
        for (int t = 0; t < T; t++) {
            int64_t v = (int64_t)dot(pw, d + t, cin, T) * m1 + bias;
            if (rw) v += (int64_t)dot(rw, x + t, cin, T) * m2;
            if (dn) {
                int64_t pre = (int64_t)dot(dn, d + t, cin, T) * b->den_m[o] + b->den_b[o];
                int32_t u = clamp32(pre >> b->den_s[o], -LUT_HALF, LUT_HALF);
                v += (int64_t)b->den_lut[u + LUT_HALF] * m3;
            }
            y[o * T + t] = (int16_t)clamp32(v >> s, 0, relu_max);
        }
    }
}

int kws_run(const kws_model_t *m, const int16_t *input, int32_t *logits, int16_t *scratch) {
    const int T = m->n_frames;
    const size_t plane = kws_scratch_elems(m) / 3;
    int16_t *h = scratch, *d = scratch + plane, *y = scratch + 2 * plane;
    const int16_t *x = input;
    for (int i = 0; i < m->n_blocks; i++) {
        const kws_block_t *b = &m->blocks[i];
        depthwise(b, x, d, T, m->act_max);
        block_out(b, x, d, y, T, m->relu_max);
        int16_t *tmp = h; h = y; y = tmp;  /* h now holds this block's output */
        x = h;
    }
    const int C = m->channels, G = m->gate_channels;
    int32_t z_sum[64];
    for (int j = 0; j < G; j++) {
        const int8_t *w = m->gate_w + j * C;
        const int64_t gm = m->gate_m[j], gb = m->gate_b[j];
        const int gs = m->gate_s[j];
        int32_t total = 0;
        for (int t = 0; t < T; t++) {
            int32_t u = clamp32(((int64_t)dot(w, x + t, C, T) * gm + gb) >> gs, -LUT_HALF, LUT_HALF);
            total += m->gate_lut[u + LUT_HALF];
        }
        z_sum[j] = total;
    }
    int best = 0;
    for (int k = 0; k < m->n_classes; k++) {
        int32_t acc = 0;
        for (int j = 0; j < G; j++) acc += (int32_t)m->fc_w[k * G + j] * z_sum[j];
        int64_t v = (int64_t)acc * m->fc_m[k] + m->fc_b[k];
        if (m->fc_den_w) {
            int32_t den = 0;
            for (int j = 0; j < G; j++) den += (int32_t)m->fc_den_w[k * G + j] * z_sum[j];
            int32_t u = clamp32(((int64_t)den * m->fc_den_m[k] + m->fc_den_b[k]) >> m->fc_den_s[k],
                                -LUT_HALF, LUT_HALF);
            v += (int64_t)m->fc_den_lut[u + LUT_HALF] * m->fc_m2[k];
        }
        logits[k] = (int32_t)(v >> m->fc_s[k]);
        if (logits[k] > logits[best]) best = k;
    }
    return best;
}
