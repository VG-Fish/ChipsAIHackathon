/* RP2040 keyword-spotting self-test firmware.
 *
 * Runs the integer SparkNet on test clips embedded in flash, checks every
 * logit against the host reference bit for bit, and prints latency.  The
 * clips are MFCCs already quantized by the host pipeline; a live deployment
 * feeds kws_run from a microphone frontend with the same input_scale.
 */
#include <stdio.h>

#include "hardware/clocks.h"
#include "pico/stdlib.h"

#include "kws_engine.h"
#include "kws_firmware_model.h" /* defines KWS_MODEL and KWS_MODEL_NAME */
#include "kws_test_vectors.h"

static int16_t scratch[KWS_SCRATCH_ELEMS];

int main(void) {
    stdio_init_all();
    const kws_model_t *m = &KWS_MODEL;
    for (;;) {
        sleep_ms(3000);
        int exact = 0, correct = 0;
        uint64_t total_us = 0, worst_us = 0;
        for (int i = 0; i < KWS_N_VECTORS; i++) {
            int32_t logits[KWS_N_CLASSES];
            uint64_t start = time_us_64();
            int predicted = kws_run(m, kws_vectors[i], logits, scratch);
            uint64_t elapsed = time_us_64() - start;
            total_us += elapsed;
            if (elapsed > worst_us) worst_us = elapsed;
            int same = 1;
            for (int k = 0; k < KWS_N_CLASSES; k++) same &= logits[k] == kws_expected[i][k];
            exact += same;
            correct += predicted == kws_labels[i];
        }
        printf("kws %s: %d/%d clips bit-exact vs host, %d/%d correct, "
               "mean %lu us, worst %lu us per clip at %lu kHz\n",
               KWS_MODEL_NAME, exact, KWS_N_VECTORS, correct, KWS_N_VECTORS,
               (unsigned long)(total_us / KWS_N_VECTORS), (unsigned long)worst_us,
               (unsigned long)(clock_get_hz(clk_sys) / 1000));
    }
}
