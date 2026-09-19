/* Host parity harness: runs kws_engine.c on quantized inputs from a file.
 *
 *   cc -O2 -DKWS_MODEL_HEADER='"kws_model.h"' -DKWS_MODEL=kws_model kws_host.c kws_engine.c
 *   ./a.out inputs.bin count logits.bin
 *
 * inputs.bin: count x [n_feat][n_frames] int16; logits.bin: count x n_classes int32 (Q16).
 */
#include <stdio.h>
#include <stdlib.h>

#include "kws_engine.h"
#include KWS_MODEL_HEADER

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s inputs.bin count logits.bin\n", argv[0]);
        return 2;
    }
    const kws_model_t *m = &KWS_MODEL;
    long count = atol(argv[2]);
    size_t clip = (size_t)m->n_feat * (size_t)m->n_frames;
    FILE *in = fopen(argv[1], "rb"), *out = fopen(argv[3], "wb");
    if (!in || !out) {
        perror("open");
        return 1;
    }
    int16_t *input = malloc(clip * sizeof(int16_t));
    int16_t *scratch = malloc(kws_scratch_elems(m) * sizeof(int16_t));
    int32_t logits[64];
    for (long i = 0; i < count; i++) {
        if (fread(input, sizeof(int16_t), clip, in) != clip) {
            fprintf(stderr, "short read at clip %ld\n", i);
            return 1;
        }
        kws_run(m, input, logits, scratch);
        fwrite(logits, sizeof(int32_t), (size_t)m->n_classes, out);
    }
    fclose(in);
    fclose(out);
    free(input);
    free(scratch);
    return 0;
}
