#include "hip/hip_runtime.h"
#include "headers/commons.h"
#include "headers/biWFA.h"
#include <cstdlib>
#include <chrono>

#define OFFSET_NULL (int32_t)(INT32_MIN/2)
#define NUM_THREADS 32
#define max_alignment_steps 10000
#define penalty_mismatch 4
#define penalty_gap_open 6
#define penalty_gap_ext 2
#define NOW std::chrono::high_resolution_clock::now();

#define gpuErrorCheck(call)                                                                     \
do{                                                                                             \
    hipError_t gpuErr = call;                                                                   \
    if(hipSuccess != gpuErr){                                                                   \
        printf("GPU Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr));     \
        exit(1);                                                                                \
    }                                                                                           \
} while(0)

// wavefront_extend_end2end_max()
__device__ void extend_max(bool *finish, const int score, int32_t *max_ak, const wf_components_t *wf, const int max_score_scope, const int alignment_k, const int32_t alignment_offset, const int pattern_len) {
    if (wf->mwavefronts[score%num_wavefronts].offsets == NULL) {
        if (wf->alignment.num_null_steps > max_score_scope) {
            *finish = true;
        } else {
            *finish = false;
        }
    } else {
        // wavefront_extend_matches_packed_end2end_max()
        bool end_reached = false;
        int32_t max_antidiag_loc = 0;
        int k;
        for (k = wf->mwavefronts[score%num_wavefronts].lo; k <= wf->mwavefronts[score%num_wavefronts].hi; ++k) {
            int32_t offset = wf->mwavefronts[score%num_wavefronts].offsets[k];
            if (offset == OFFSET_NULL) {continue;}

            // wavefront_extend_matches_kernel_blockwise() or wavefront_extend_matches_kernel()
            int equal_chars = 0;
            for (int i = offset; i < pattern_len; i++) {
                if((i - k) >= 0 && (i - k) < pattern_len) {
                    if (wf->alignment.pattern[i - k] == wf->alignment.text[i]) {
                        equal_chars++;
                    } else break;
                }
            }
            offset += equal_chars;

            // Return extended offset
            wf->mwavefronts[score%num_wavefronts].offsets[k] = offset;

            int32_t antidiag = (2 * wf->mwavefronts[score%num_wavefronts].offsets[k]) - k;
            if (max_antidiag_loc < antidiag) {
                max_antidiag_loc = antidiag;
            }
        }

        *max_ak = max_antidiag_loc;

        // wavefront_termination_end2end()
        // End component matrixM
        if (wf->mwavefronts[score%num_wavefronts].lo > alignment_k || alignment_k > wf->mwavefronts[score%num_wavefronts].hi) {
            end_reached = false;
        } else {
            int32_t moffset = wf->mwavefronts[score%num_wavefronts].offsets[alignment_k];
            if (moffset < alignment_offset) {
                end_reached = false;
            } else {
                end_reached = true;
            }
        }

        if (end_reached) {
            *finish = true;
        } else {
            *finish = false;
        }
    }
}

// wavefront_extend_end2end()
__device__ void extend(bool *finish, const int score, const wf_components_t *wf, const int max_score_scope, const int alignment_k, const int32_t alignment_offset, const int pattern_len) {
    if (wf->mwavefronts[score%num_wavefronts].offsets == NULL) {
        if (wf->alignment.num_null_steps > max_score_scope) {
            *finish = true;
        } else {
            *finish = false;
        }
    } else {
        // wavefront_extend_matches_packed_end2end()
        bool end_reached = false;
        int k;
        for (k = wf->mwavefronts[score%num_wavefronts].lo; k <= wf->mwavefronts[score%num_wavefronts].hi; ++k) {
            int32_t offset = wf->mwavefronts[score%num_wavefronts].offsets[k];
            if (offset == OFFSET_NULL) {continue;}

            // wavefront_extend_matches_kernel_blockwise() or wavefront_extend_matches_kernel()
            int equal_chars = 0;

            for (int i = offset; i < pattern_len; i++) {
                if((i - k) >= 0 && (i - k) < pattern_len) {
                    if (wf->alignment.pattern[i - k] == wf->alignment.text[i]) {
                        equal_chars++;
                    } else break;
                }
            }
            offset += equal_chars;

            // Return extended offset
            wf->mwavefronts[score%num_wavefronts].offsets[k] = offset;
        }

        // wavefront_termination_end2end()
        // End component matrixM
        if (wf->mwavefronts[score%num_wavefronts].lo > alignment_k || alignment_k > wf->mwavefronts[score%num_wavefronts].hi) {
            end_reached = false;
        } else {
            int32_t moffset = wf->mwavefronts[score%num_wavefronts].offsets[alignment_k];
            if (moffset < alignment_offset) {
                end_reached = false;
            } else {
                end_reached = true;
            }
        }
        if (end_reached) {
            *finish = true;
        } else {
            *finish = false;
        }
    }
}

__device__ void nextWF(int *score, wf_components_t *wf, const bool forward, const int max_score_scope, const int text_len, const int pattern_len, int32_t *matrix_wf_m_g, int32_t *matrix_wf_i_g, int32_t *matrix_wf_d_g) {
    // Compute next (s+1) wavefront
    ++(*score);

    int score_mod = *score%num_wavefronts;

    // wavefront_compute_affine()
    int mismatch = *score - penalty_mismatch;
    int gap_open = *score - penalty_gap_open - penalty_gap_ext;
    int gap_extend = *score - penalty_gap_ext;

    // wavefront_compute_get_mwavefront()
    if((*score / num_wavefronts) > 0) {
        // Resetting old wavefronts' values
        wf->mwavefronts[score_mod].lo = -1;
        wf->mwavefronts[score_mod].hi = 1;
        wf->iwavefronts[score_mod].lo = -1;
        wf->iwavefronts[score_mod].hi = 1;
        wf->dwavefronts[score_mod].lo = -1;
        wf->dwavefronts[score_mod].hi = 1;
    }
    wf->mwavefronts[score_mod].offsets = matrix_wf_m_g + (num_wavefronts * wf_length * hipBlockIdx_x) + (score_mod * wf_length) + wf_length/2;
    wf->mwavefronts[score_mod].null = false;
    wf->iwavefronts[score_mod].offsets = matrix_wf_i_g + (num_wavefronts * wf_length * hipBlockIdx_x) + (score_mod * wf_length) + wf_length/2;
    wf->iwavefronts[score_mod].null = false;
    wf->dwavefronts[score_mod].offsets = matrix_wf_d_g + (num_wavefronts * wf_length * hipBlockIdx_x) + (score_mod * wf_length) + wf_length/2;
    wf->dwavefronts[score_mod].null = false;

    wf_t in_mwavefront_misms = (mismatch < 0 || wf->mwavefronts[mismatch%num_wavefronts].offsets == NULL || wf->mwavefronts[mismatch%num_wavefronts].null) ? wf->wavefront_null : wf->mwavefronts[mismatch%num_wavefronts];
    wf_t in_mwavefront_open = (gap_open < 0 || wf->mwavefronts[gap_open%num_wavefronts].offsets == NULL || wf->mwavefronts[gap_open%num_wavefronts].null) ? wf->wavefront_null : wf->mwavefronts[gap_open%num_wavefronts];
    wf_t in_iwavefront_ext = (gap_extend < 0 || wf->iwavefronts[gap_extend%num_wavefronts].offsets == NULL || wf->iwavefronts[gap_extend%num_wavefronts].null) ? wf->wavefront_null : wf->iwavefronts[gap_extend%num_wavefronts];
    wf_t in_dwavefront_ext = (gap_extend < 0 || wf->dwavefronts[gap_extend%num_wavefronts].offsets == NULL || wf->dwavefronts[gap_extend%num_wavefronts].null) ? wf->wavefront_null : wf->dwavefronts[gap_extend%num_wavefronts];

    if (in_mwavefront_misms.null && in_mwavefront_open.null && in_iwavefront_ext.null && in_dwavefront_ext.null) {
        // wavefront_compute_allocate_output_null()
        wf->alignment.num_null_steps++; // Increment null-steps
        // Nullify Wavefronts
        wf->mwavefronts[score_mod].null = true;
        wf->iwavefronts[score_mod].null = true;
        wf->dwavefronts[score_mod].null = true;
    } else {
        wf->alignment.num_null_steps = 0;
        int hi, lo;

        // wavefront_compute_limits_input()
        int min_lo = in_mwavefront_misms.lo;
        int max_hi = in_mwavefront_misms.hi;

        if (!in_mwavefront_open.null && min_lo > (in_mwavefront_open.lo - 1)) min_lo = in_mwavefront_open.lo - 1;
        if (!in_mwavefront_open.null && max_hi < (in_mwavefront_open.hi + 1)) max_hi = in_mwavefront_open.hi + 1;
        if (!in_iwavefront_ext.null && min_lo > (in_iwavefront_ext.lo + 1)) min_lo = in_iwavefront_ext.lo + 1;
        if (!in_iwavefront_ext.null && max_hi < (in_iwavefront_ext.hi + 1)) max_hi = in_iwavefront_ext.hi + 1;
        if (!in_dwavefront_ext.null && min_lo > (in_dwavefront_ext.lo - 1)) min_lo = in_dwavefront_ext.lo - 1;
        if (!in_dwavefront_ext.null && max_hi < (in_dwavefront_ext.hi - 1)) max_hi = in_dwavefront_ext.hi - 1;
        lo = min_lo;
        hi = max_hi;

        // wavefront_compute_allocate_output()
        int effective_lo = lo;
        int effective_hi = hi;

        // wavefront_compute_limits_output()
        int eff_lo = effective_lo - (max_score_scope + 1);
        int eff_hi = effective_hi + (max_score_scope + 1);
        effective_lo = MIN(eff_lo, wf->alignment.historic_min_lo);
        effective_hi = MAX(eff_hi, wf->alignment.historic_max_hi);
        wf->alignment.historic_min_lo = effective_lo;
        wf->alignment.historic_max_hi = effective_hi;

        // Allocate M-Wavefront
        wf->mwavefronts[score_mod].lo = lo;
        wf->mwavefronts[score_mod].hi = hi;
        // Allocate I1-Wavefront
        if (!in_mwavefront_open.null || !in_iwavefront_ext.null) {
            wf->iwavefronts[score_mod].lo = lo;
            wf->iwavefronts[score_mod].hi = hi;
        } else {
            wf->iwavefronts[score_mod].null = true;
        }
        // Allocate D1-Wavefront
        if (!in_mwavefront_open.null || !in_dwavefront_ext.null) {
            wf->dwavefronts[score_mod].lo = lo;
            wf->dwavefronts[score_mod].hi = hi;
        } else {
            wf->dwavefronts[score_mod].null = true;
        }

        // wavefront_compute_init_ends()
        // Init wavefront ends
        bool m_misms_null = in_mwavefront_misms.null;
        bool m_gap_null = in_mwavefront_open.null;
        bool i_ext_null = in_iwavefront_ext.null;
        bool d_ext_null = in_dwavefront_ext.null;

        if (!m_misms_null) {
            // wavefront_compute_init_ends_wf_higher()
            if (in_mwavefront_misms.wf_elements_init_max >= hi) {
            } else {
                // Initialize lower elements
                int max_init = MAX(in_mwavefront_misms.wf_elements_init_max, in_mwavefront_misms.hi);
                int k;
                for (k = max_init + 1; k <= hi; ++k) {
                    in_mwavefront_misms.offsets[k] = OFFSET_NULL;
                }
                // Set new maximum
                in_mwavefront_misms.wf_elements_init_max = hi;
            }
            // wavefront_compute_init_ends_wf_lower()
            if (in_mwavefront_misms.wf_elements_init_min <= lo) {
            } else {
                // Initialize lower elements
                int min_init = MIN(in_mwavefront_misms.wf_elements_init_min, in_mwavefront_misms.lo);
                int k;
                for (k = lo; k < min_init; ++k) {
                    in_mwavefront_misms.offsets[k] = OFFSET_NULL;
                }
                // Set new minimum
                in_mwavefront_misms.wf_elements_init_min = lo;
            }
        }
        if (!m_gap_null) {
            // wavefront_compute_init_ends_wf_higher()
            if (in_mwavefront_open.wf_elements_init_max >= hi + 1) {
            } else {
                // Initialize lower elements
                int max_init = MAX(in_mwavefront_open.wf_elements_init_max, in_mwavefront_open.hi);
                int k;
                for (k = max_init + 1; k <= hi + 1; ++k) {
                    in_mwavefront_open.offsets[k] = OFFSET_NULL;
                }
                // Set new maximum
                in_mwavefront_open.wf_elements_init_max = hi + 1;
            }
            // wavefront_compute_init_ends_wf_lower()
            if (in_mwavefront_open.wf_elements_init_min <= lo - 1) {
            } else {
                // Initialize lower elements
                int min_init = MIN(in_mwavefront_open.wf_elements_init_min, in_mwavefront_open.lo);
                int k;
                for (k = lo - 1; k < min_init; ++k) {
                    in_mwavefront_open.offsets[k] = OFFSET_NULL;
                }
                // Set new minimum
                in_mwavefront_open.wf_elements_init_min = lo - 1;
            }
        }
        if (!i_ext_null) {
            // wavefront_compute_init_ends_wf_higher()
            if (in_iwavefront_ext.wf_elements_init_max >= hi) {
            } else {
                // Initialize lower elements
                int max_init = MAX(in_iwavefront_ext.wf_elements_init_max, in_iwavefront_ext.hi);
                int k;
                for (k = max_init + 1; k <= hi; ++k) {
                    in_iwavefront_ext.offsets[k] = OFFSET_NULL;
                }
                // Set new maximum
                in_iwavefront_ext.wf_elements_init_max = hi;
            }
            // wavefront_compute_init_ends_wf_lower()
            if (in_iwavefront_ext.wf_elements_init_min <= lo - 1) {
            } else {
                // Initialize lower elements
                int min_init = MIN(in_iwavefront_ext.wf_elements_init_min, in_iwavefront_ext.lo);
                int k;
                for (k = lo - 1; k < min_init; ++k) {
                    in_iwavefront_ext.offsets[k] = OFFSET_NULL;
                }
                // Set new minimum
                in_iwavefront_ext.wf_elements_init_min = lo - 1;
            }
        }
        if (!d_ext_null) {
            // wavefront_compute_init_ends_wf_higher()
            if (in_dwavefront_ext.wf_elements_init_max >= hi + 1) {
            } else {
                // Initialize lower elements
                int max_init = MAX(in_dwavefront_ext.wf_elements_init_max, in_dwavefront_ext.hi);
                int k;
                for (k = max_init + 1; k <= hi + 1; ++k) {
                    in_dwavefront_ext.offsets[k] = OFFSET_NULL;
                }
                // Set new maximum
                in_dwavefront_ext.wf_elements_init_max = hi + 1;
            }
            // wavefront_compute_init_ends_wf_lower()
            if (in_dwavefront_ext.wf_elements_init_min <= lo) {
            } else {
                // Initialize lower elements
                int min_init = MIN(in_dwavefront_ext.wf_elements_init_min, in_dwavefront_ext.lo);
                int k;
                for (k = lo; k < min_init; ++k) {
                    in_dwavefront_ext.offsets[k] = OFFSET_NULL;
                }
                // Set new minimum
                in_dwavefront_ext.wf_elements_init_min = lo;
            }
        }

        //wavefront_compute_affine_idm()
        // Compute-Next kernel loop
        int tidx = hipThreadIdx_x;
        for (int i = lo; i <= hi; i += hipBlockDim_x) {
            int idx = tidx + i;
            if (idx <= hi) {
                // Update I1
                int32_t ins_o = in_mwavefront_open.offsets[idx - 1];
                int32_t ins_e = in_iwavefront_ext.offsets[idx - 1];
                int32_t ins = MAX(ins_o, ins_e) + 1;
                wf->iwavefronts[score_mod].offsets[idx] = ins;

                // Update D1
                int32_t del_o = in_mwavefront_open.offsets[idx + 1];
                int32_t del_e = in_dwavefront_ext.offsets[idx + 1];
                int32_t del = MAX(del_o, del_e);
                wf->dwavefronts[score_mod].offsets[idx] = del;

                // Update M
                int32_t misms = in_mwavefront_misms.offsets[idx] + 1;
                int32_t max = MAX(del, MAX(misms, ins));

                // Adjust offset out of boundaries
                uint32_t h = max;
                uint32_t v = max - idx;
                if (h > text_len) max = OFFSET_NULL;
                if (v > pattern_len) max = OFFSET_NULL;
                wf->mwavefronts[score_mod].offsets[idx] = max;
            }
        }

        // wavefront_compute_process_ends()
        if (wf->mwavefronts[score_mod].offsets) {
            // wavefront_compute_trim_ends()
            int k;
            int lo = wf->mwavefronts[score_mod].lo;
            for (k = wf->mwavefronts[score_mod].hi; k >= lo; --k) {
                // Fetch offset
                int32_t offset = wf->mwavefronts[score_mod].offsets[k];
                // Check boundaries
                uint32_t h = offset; // Make unsigned to avoid checking negative
                uint32_t v = offset - k; // Make unsigned to avoid checking negative
                if (h <= text_len && v <= pattern_len) break;
            }
            wf->mwavefronts[score_mod].hi = k; // Set new hi
            wf->mwavefronts[score_mod].wf_elements_init_max = k;
            // Trim from lo
            int hi = wf->mwavefronts[score_mod].hi;
            for (k = wf->mwavefronts[score_mod].lo ; k <= hi; ++k) {
                // Fetch offset
                int32_t offset = wf->mwavefronts[score_mod].offsets[k];
                // Check boundaries
                uint32_t h = offset; // Make unsigned to avoid checking negative
                uint32_t v = offset - k; // Make unsigned to avoid checking negative
                if (h <= text_len && v <= pattern_len) break;
            }
            wf->mwavefronts[score_mod].lo = k; // Set new lo
            wf->mwavefronts[score_mod].wf_elements_init_min = k;
            wf->mwavefronts[score_mod].null = (wf->mwavefronts[score_mod].lo > wf->mwavefronts[score_mod].hi);
        }
        if (wf->iwavefronts[score_mod].offsets) {
            // wavefront_compute_trim_ends()
            int k;
            int lo = wf->iwavefronts[score_mod].lo;
            for (k = wf->iwavefronts[score_mod].hi; k >= lo; --k) {
                // Fetch offset
                int32_t offset = wf->iwavefronts[score_mod].offsets[k];
                // Check boundaries
                uint32_t h = offset; // Make unsigned to avoid checking negative
                uint32_t v = offset - k; // Make unsigned to avoid checking negative
                if (h <= text_len && v <= pattern_len) break;
            }
            wf->iwavefronts[score_mod].hi = k; // Set new hi
            wf->iwavefronts[score_mod].wf_elements_init_max = k;
            // Trim from lo
            int hi = wf->iwavefronts[score_mod].hi;
            for (k = wf->iwavefronts[score_mod].lo; k <= hi; ++k) {
                // Fetch offset
                int32_t offset = wf->iwavefronts[score_mod].offsets[k];
                // Check boundaries
                uint32_t h = offset; // Make unsigned to avoid checking negative
                uint32_t v = offset - k; // Make unsigned to avoid checking negative
                if (h <= text_len && v <= pattern_len) break;
            }
            wf->iwavefronts[score_mod].lo = k; // Set new lo
            wf->iwavefronts[score_mod].wf_elements_init_min = k;
            wf->iwavefronts[score_mod].null = (wf->iwavefronts[score_mod].lo > wf->iwavefronts[score_mod].hi);
        }
        if (wf->dwavefronts[score_mod].offsets) {
            // wavefront_compute_trim_ends()
            int k;
            int lo = wf->dwavefronts[score_mod].lo;
            for (k = wf->dwavefronts[score_mod].hi; k >= lo ; --k) {
                // Fetch offset
                int32_t offset = wf->dwavefronts[score_mod].offsets[k];
                // Check boundaries
                uint32_t h = offset; // Make unsigned to avoid checking negative
                uint32_t v = offset - k; // Make unsigned to avoid checking negative
                if (h <= text_len && v <= pattern_len) break;
            }
            wf->dwavefronts[score_mod].hi = k; // Set new hi
            wf->dwavefronts[score_mod].wf_elements_init_max = k;
            // Trim from lo
            int hi = wf->dwavefronts[score_mod].hi;
            for (k = wf->dwavefronts[score_mod].lo; k <= hi; ++k) {
                // Fetch offset
                int32_t offset = wf->dwavefronts[score_mod].offsets[k];
                // Check boundaries
                uint32_t h = offset; // Make unsigned to avoid checking negative
                uint32_t v = offset - k; // Make unsigned to avoid checking negative
                if (h <= text_len && v <= pattern_len) break;
            }
            wf->dwavefronts[score_mod].lo = k; // Set new lo
            wf->dwavefronts[score_mod].wf_elements_init_min = k;
            wf->dwavefronts[score_mod].null = (wf->dwavefronts[score_mod].lo > wf->dwavefronts[score_mod].hi);
        }
    }
}

// wavefront_bialign_breakpoint_indel2indel()
__device__ void breakpoint_indel2indel(const int score_0, const int score_1, const wf_t *dwf_0, const wf_t *dwf_1, int *breakpoint_score, const int text_len, const int pattern_len) {
    int lo_0 = dwf_0->lo;
    int hi_0 = dwf_0->hi;
    int lo_1 = text_len - pattern_len - dwf_1->hi;
    int hi_1 = text_len - pattern_len - dwf_1->lo;

    if (hi_1 < lo_0 || hi_0 < lo_1) return;
    // Compute overlapping interval
    int min_hi = MIN(hi_0, hi_1);
    int max_lo = MAX(lo_0, lo_1);
    int k_0;
    for (k_0 = max_lo; k_0 <= min_hi; k_0++) {
        int k_1 = text_len - pattern_len - k_0;
        // Fetch offsets
        int dh_0 = dwf_0->offsets[k_0];
        int dh_1 = dwf_1->offsets[k_1];
        // Check breakpoint d2d
        if (dh_0 + dh_1 >= text_len && score_0 + score_1 - penalty_gap_open < *breakpoint_score) {
            *breakpoint_score = score_0 + score_1 - penalty_gap_open;
            return;
        }
    }
}

// wavefront_bialign_breakpoint_m2m()
__device__ void breakpoint_m2m(const int score_0, const int score_1, const wf_t *mwf_0, const wf_t *mwf_1, int *breakpoint_score, const int text_len, const int pattern_len) {
    // Check wavefronts overlapping
    int lo_0 = mwf_0->lo;
    int hi_0 = mwf_0->hi;
    int lo_1 = text_len - pattern_len - mwf_1->hi;
    int hi_1 = text_len - pattern_len - mwf_1->lo;
    if (hi_1 < lo_0 || hi_0 < lo_1) return;
    // Compute overlapping interval
    int min_hi = MIN(hi_0, hi_1);
    int max_lo = MAX(lo_0, lo_1);
    int k_0;
    for (k_0 = max_lo; k_0 <= min_hi; k_0++) {
        const int k_1 = text_len - pattern_len - k_0;
        // Fetch offsets
        const int mh_0 = mwf_0->offsets[k_0];
        const int mh_1 = mwf_1->offsets[k_1];
        // Check breakpoint m2m
        if (mh_0 + mh_1 >= text_len && score_0 + score_1 < *breakpoint_score) {
            *breakpoint_score = score_0 + score_1;
            return;
        }
    }
}

// wavefront_bialign_overlap()
__device__ void overlap(const int score_0, const wf_components_t *wf_0, const int score_1, const wf_components_t *wf_1, const int max_score_scope, int *breakpoint_score, const int text_len, const int pattern_len) {
    // Fetch wavefront-0
    int score_mod_0 = score_0%num_wavefronts;
    wf_t *mwf_0 = &wf_0->mwavefronts[score_mod_0];

    if (mwf_0 == NULL) return;
    wf_t *d1wf_0 = &wf_0->dwavefronts[score_mod_0];
    wf_t *i1wf_0 = &wf_0->iwavefronts[score_mod_0];

    // Traverse all scores-1
    int i;
    for (i = 0; i < max_score_scope; ++i) {
        // Compute score
        const int score_i = score_1 - i;
        if (score_i < 0) break;
        int score_mod_i = score_i%num_wavefronts;

        if (score_0 + score_i - penalty_gap_open >= *breakpoint_score) continue;
        // Check breakpoint d2d
        wf_t *d1wf_1 = &wf_1->dwavefronts[score_mod_i];
        if (d1wf_0 != NULL && d1wf_1 != NULL) {
            breakpoint_indel2indel(score_0, score_i, d1wf_0, d1wf_1, breakpoint_score, text_len, pattern_len);
        }
        // Check breakpoint i2i
        wf_t *i1wf_1 = &wf_1->iwavefronts[score_mod_i];
        if (i1wf_0 != NULL && i1wf_1 != NULL) {
            breakpoint_indel2indel(score_0, score_i, i1wf_0, i1wf_1, breakpoint_score, text_len, pattern_len);
        }
        // Check M-breakpoints (indel, edit, gap-linear)
        if (score_0 + score_i >= *breakpoint_score) continue;
        wf_t *mwf_1 = &wf_1->mwavefronts[score_mod_i];
        if (mwf_1 != NULL) {
            breakpoint_m2m(score_0, score_i, mwf_0, mwf_1, breakpoint_score, text_len, pattern_len);
        }
    }
}

__global__ void biWFA(char *pattern_f_g, char *text_f_g, char *pattern_r_g, char *text_r_g, int *breakpoint_score_g, wf_t *mwavefronts_f_g,
                      wf_t *iwavefronts_f_g, wf_t *dwavefronts_f_g, wf_t *mwavefronts_r_g, wf_t *iwavefronts_r_g, wf_t *dwavefronts_r_g,
                      const int lo_g, const int hi_g, int32_t *offsets_g, const int pattern_len, const int text_len, const int max_score_scope, int32_t *matrix_wf_m_f_g,
                      int32_t *matrix_wf_i_f_g, int32_t *matrix_wf_d_f_g, int32_t *matrix_wf_m_r_g, int32_t *matrix_wf_i_r_g, int32_t *matrix_wf_d_r_g) {
    int lo = lo_g;
    int hi = hi_g;

    // Wavefronts initialization
    for (int i = 0; i < num_wavefronts * wf_length; i += hipBlockDim_x) {
        if (i + hipThreadIdx_x < num_wavefronts * wf_length) {
            *(matrix_wf_m_f_g + (num_wavefronts * wf_length * hipBlockIdx_x) + (i + hipThreadIdx_x)) = OFFSET_NULL;
            *(matrix_wf_i_f_g + (num_wavefronts * wf_length * hipBlockIdx_x) + (i + hipThreadIdx_x)) = OFFSET_NULL;
            *(matrix_wf_d_f_g + (num_wavefronts * wf_length * hipBlockIdx_x) + (i + hipThreadIdx_x)) = OFFSET_NULL;
            *(matrix_wf_m_r_g + (num_wavefronts * wf_length * hipBlockIdx_x) + (i + hipThreadIdx_x)) = OFFSET_NULL;
            *(matrix_wf_i_r_g + (num_wavefronts * wf_length * hipBlockIdx_x) + (i + hipThreadIdx_x)) = OFFSET_NULL;
            *(matrix_wf_d_r_g + (num_wavefronts * wf_length * hipBlockIdx_x) + (i + hipThreadIdx_x)) = OFFSET_NULL;
        }
    }
    for (int i = 0; i < num_wavefronts; i += hipBlockDim_x) {
        if (i + hipThreadIdx_x < num_wavefronts) {
            (mwavefronts_f_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->null = true;
            (mwavefronts_f_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->lo = 0;
            (mwavefronts_f_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->hi = 0;
            (mwavefronts_f_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->offsets = NULL;
            (mwavefronts_f_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->wf_elements_init_max = 0;
            (mwavefronts_f_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->wf_elements_init_min = 0;
            (mwavefronts_r_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->null = true;
            (mwavefronts_r_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->lo = 0;
            (mwavefronts_r_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->hi = 0;
            (mwavefronts_r_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->offsets = NULL;
            (mwavefronts_r_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->wf_elements_init_max = 0;
            (mwavefronts_r_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->wf_elements_init_min = 0;

            (iwavefronts_f_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->null = true;
            (iwavefronts_f_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->lo = 0;
            (iwavefronts_f_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->hi = 0;
            (iwavefronts_f_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->offsets = NULL;
            (iwavefronts_f_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->wf_elements_init_max = 0;
            (iwavefronts_f_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->wf_elements_init_min = 0;
            (iwavefronts_r_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->null = true;
            (iwavefronts_r_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->lo = 0;
            (iwavefronts_r_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->hi = 0;
            (iwavefronts_r_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->offsets = NULL;
            (iwavefronts_r_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->wf_elements_init_max = 0;
            (iwavefronts_r_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->wf_elements_init_min = 0;

            (dwavefronts_f_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->null = true;
            (dwavefronts_f_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->lo = 0;
            (dwavefronts_f_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->hi = 0;
            (dwavefronts_f_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->offsets = NULL;
            (dwavefronts_f_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->wf_elements_init_max = 0;
            (dwavefronts_f_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->wf_elements_init_min = 0;
            (dwavefronts_r_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->null = true;
            (dwavefronts_r_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->lo = 0;
            (dwavefronts_r_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->hi = 0;
            (dwavefronts_r_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->offsets = NULL;
            (dwavefronts_r_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->wf_elements_init_max = 0;
            (dwavefronts_r_g + (num_wavefronts * hipBlockIdx_x) + (i + hipThreadIdx_x))->wf_elements_init_min = 0;
        }
    }
    __syncthreads();

    wf_components_t wf_f, wf_r;
    wf_alignment_t alignment_f, alignment_r;
    int max_antidiag, score_f, score_r, forward_max_ak, reverse_max_ak, breakpoint_score, alignment_k;
    bool finish;

    // Forward wavefront
    alignment_f.pattern = pattern_f_g + (pattern_len * hipBlockIdx_x);
    alignment_f.text = text_f_g + (text_len * hipBlockIdx_x);

    // wavefront_components_dimensions()
    alignment_f.historic_max_hi = 0;
    alignment_f.historic_min_lo = 0;
    wf_f.alignment = alignment_f;

    // Reverse wavefront
    alignment_r.pattern = pattern_r_g + (pattern_len * hipBlockIdx_x);
    alignment_r.text = text_r_g + (text_len * hipBlockIdx_x);

    // wavefront_components_dimensions()
    alignment_r.historic_max_hi = 0;
    alignment_r.historic_min_lo = 0;
    wf_r.alignment = alignment_r;

    // wavefront_aligner_init()
    wf_f.alignment.num_null_steps = 0;
    wf_f.alignment.historic_max_hi = hi;
    wf_f.alignment.historic_min_lo = lo;
    wf_r.alignment.num_null_steps = 0;
    wf_r.alignment.historic_max_hi = hi;
    wf_r.alignment.historic_min_lo = lo;

    // wavefront_components_allocate_wf()
    // Forward wavefront
    wf_f.mwavefronts = mwavefronts_f_g + (num_wavefronts * hipBlockIdx_x);
    wf_f.iwavefronts = iwavefronts_f_g + (num_wavefronts * hipBlockIdx_x);
    wf_f.dwavefronts = dwavefronts_f_g + (num_wavefronts * hipBlockIdx_x);

    wf_f.mwavefronts[0].offsets = matrix_wf_m_f_g + (num_wavefronts * wf_length * hipBlockIdx_x) + 0*wf_length + wf_length/2;
    wf_f.iwavefronts[0].offsets = matrix_wf_i_f_g + (num_wavefronts * wf_length * hipBlockIdx_x) + 0*wf_length + wf_length/2;
    wf_f.dwavefronts[0].offsets = matrix_wf_d_f_g + (num_wavefronts * wf_length * hipBlockIdx_x) + 0*wf_length + wf_length/2;

    // wavefront_init()
    wf_f.mwavefronts[0].null = false;
    wf_f.mwavefronts[0].lo = -1;
    wf_f.mwavefronts[0].hi = 1;
    wf_f.mwavefronts[0].offsets[0] = 0;
    wf_f.mwavefronts[0].wf_elements_init_min = 0;
    wf_f.mwavefronts[0].wf_elements_init_max = 0;

    wf_f.iwavefronts[0].null = true;
    wf_f.iwavefronts[0].lo = -1;
    wf_f.iwavefronts[0].hi = 1;
    wf_f.iwavefronts[0].wf_elements_init_min = 0;
    wf_f.iwavefronts[0].wf_elements_init_max = 0;

    wf_f.dwavefronts[0].null = true;
    wf_f.dwavefronts[0].lo = -1;
    wf_f.dwavefronts[0].hi = 1;
    wf_f.dwavefronts[0].wf_elements_init_min = 0;
    wf_f.dwavefronts[0].wf_elements_init_max = 0;

    // wavefront_init_null()
    wf_f.wavefront_null.null = true;
    wf_f.wavefront_null.lo = 1;
    wf_f.wavefront_null.hi = -1;
    wf_f.wavefront_null.offsets = offsets_g + wf_length/2;
    wf_f.wavefront_null.wf_elements_init_min = 0;
    wf_f.wavefront_null.wf_elements_init_max = 0;

    // wavefront_components_allocate_wf()
    // Reverse wavefront
    wf_r.mwavefronts = mwavefronts_r_g + (num_wavefronts * hipBlockIdx_x);
    wf_r.iwavefronts = iwavefronts_r_g + (num_wavefronts * hipBlockIdx_x);
    wf_r.dwavefronts = dwavefronts_r_g + (num_wavefronts * hipBlockIdx_x);
    wf_r.mwavefronts[0].offsets = matrix_wf_m_r_g + (num_wavefronts * wf_length * hipBlockIdx_x) + 0*wf_length + wf_length/2;
    wf_r.iwavefronts[0].offsets = matrix_wf_i_r_g + (num_wavefronts * wf_length * hipBlockIdx_x) + 0*wf_length + wf_length/2;
    wf_r.dwavefronts[0].offsets = matrix_wf_d_r_g + (num_wavefronts * wf_length * hipBlockIdx_x) + 0*wf_length + wf_length/2;

    wf_r.mwavefronts[0].null = false;
    wf_r.mwavefronts[0].lo = -1;
    wf_r.mwavefronts[0].hi = 1;
    wf_r.mwavefronts[0].offsets[0] = 0;
    wf_r.mwavefronts[0].wf_elements_init_min = 0;
    wf_r.mwavefronts[0].wf_elements_init_max = 0;

    wf_r.iwavefronts[0].null = true;
    wf_r.iwavefronts[0].lo = -1;
    wf_r.iwavefronts[0].hi = 1;
    wf_r.iwavefronts[0].wf_elements_init_min = 0;
    wf_r.iwavefronts[0].wf_elements_init_max = 0;

    wf_r.dwavefronts[0].null = true;
    wf_r.dwavefronts[0].lo = -1;
    wf_r.dwavefronts[0].hi = 1;
    wf_r.dwavefronts[0].wf_elements_init_min = 0;
    wf_r.dwavefronts[0].wf_elements_init_max = 0;

    // wavefront_init_null()
    wf_r.wavefront_null.null = true;
    wf_r.wavefront_null.lo = 1;
    wf_r.wavefront_null.hi = -1;
    wf_r.wavefront_null.offsets = offsets_g + wf_length/2;
    wf_r.wavefront_null.wf_elements_init_min = 0;
    wf_r.wavefront_null.wf_elements_init_max = 0;

    // wavefront_bialign_find_breakpoint()
    max_antidiag = text_len + pattern_len - 1;
    score_f = 0;
    score_r = 0;
    forward_max_ak = 0;
    reverse_max_ak = 0;

    // Prepare and perform first bialignment step
    breakpoint_score = INT_MAX;

    finish = false;
    alignment_k = text_len - pattern_len;

    extend_max(&finish, score_f, &forward_max_ak, &wf_f, max_score_scope, alignment_k, (int32_t)text_len, pattern_len);
    if(finish) finish = true;
    extend_max(&finish, score_r, &reverse_max_ak, &wf_r, max_score_scope, alignment_k, (int32_t)text_len, pattern_len);
    if(finish) finish = true;

    if(finish) return;

    // Compute wavefronts of increasing score until both wavefronts overlap
    __shared__ int max_ak;
    __shared__ bool last_wf_forward;
    max_ak = 0;
    last_wf_forward = false;
    while (true) {
        if (forward_max_ak + reverse_max_ak >= max_antidiag) break;
        nextWF(&score_f, &wf_f, true, max_score_scope, text_len, pattern_len, matrix_wf_m_f_g, matrix_wf_i_f_g, matrix_wf_d_f_g);
        extend_max(&finish, score_f, &max_ak, &wf_f, max_score_scope, alignment_k, (int32_t) text_len, pattern_len);
        if (forward_max_ak < max_ak) forward_max_ak = max_ak;
        last_wf_forward = true;
        if (forward_max_ak + reverse_max_ak >= max_antidiag) break;
        nextWF(&score_r, &wf_r, false, max_score_scope, text_len, pattern_len, matrix_wf_m_r_g, matrix_wf_i_r_g, matrix_wf_d_r_g);
        extend_max(&finish, score_r, &max_ak, &wf_r, max_score_scope, alignment_k, (int32_t) text_len, pattern_len);
        if (reverse_max_ak < max_ak) reverse_max_ak = max_ak;
        last_wf_forward = false;
    }

    // Advance until overlap is found
    __shared__ int min_score_f, min_score_r;
    while (true) {
        if (last_wf_forward) {
            // Check overlapping wavefronts
            min_score_r = (score_r > max_score_scope - 1) ? score_r - (max_score_scope - 1) : 0;
            if (score_f + min_score_r - penalty_gap_open >= breakpoint_score) break;
            overlap(score_f, &wf_f, score_r, &wf_r, max_score_scope, &breakpoint_score, text_len, pattern_len);
            nextWF(&score_r, &wf_r, true, max_score_scope, text_len, pattern_len, matrix_wf_m_r_g, matrix_wf_i_r_g, matrix_wf_d_r_g);
            extend(&finish, score_r, &wf_r, max_score_scope, alignment_k, (int32_t) text_len, pattern_len);
        }
        // Check overlapping wavefronts
        min_score_f = (score_f > max_score_scope - 1) ? score_f - (max_score_scope - 1) : 0;
        if (min_score_f + score_r - penalty_gap_open >= breakpoint_score) break;
        overlap(score_r, &wf_r, score_f, &wf_f, max_score_scope, &breakpoint_score, text_len, pattern_len);
        nextWF(&score_f, &wf_f, false, max_score_scope, text_len, pattern_len, matrix_wf_m_f_g, matrix_wf_i_f_g, matrix_wf_d_f_g);
        extend(&finish, score_f, &wf_f, max_score_scope, alignment_k, (int32_t) text_len, pattern_len);

        if (score_r + score_f >= max_alignment_steps) break;
        // Enable always
        last_wf_forward = true;
    }

    breakpoint_score = -breakpoint_score;
    *(breakpoint_score_g + hipBlockIdx_x) = breakpoint_score;
}

// wavefront_bialign() | wavefront_bialign_compute_score()
int main(int argc, char *argv[]) {
    if(argc < 2 || argc > 3) {
        printf("Wrong number of parameters!\n");
        return 1;
    }

    int s_flag = 0;
    if(argc == 3 && strcmp(argv[2], "-s") == 0) {
        s_flag = 1;
    }

    FILE *fp = fopen(argv[1], "r");
    if(fp == NULL) {
        printf("Cannot open file!\n");
        return 1;
    }

    int num_alignments, pattern_len, text_len;
    fscanf(fp, "%d", &num_alignments);
    fscanf(fp, "%d", &pattern_len);
    fscanf(fp, "%d", &text_len);

    int *breakpoint_score, *breakpoint_score_g;
    char *pattern_f, *text_f, *pattern_r, *text_r;
    char *pattern_f_g, *text_f_g, *pattern_r_g, *text_r_g;
    wf_t *mwavefronts_f_g, *iwavefronts_f_g, *dwavefronts_f_g;
    wf_t *mwavefronts_r_g, *iwavefronts_r_g, *dwavefronts_r_g;
    int32_t *matrix_wf_m_f_g, *matrix_wf_i_f_g, *matrix_wf_d_f_g, *matrix_wf_m_r_g, *matrix_wf_i_r_g, *matrix_wf_d_r_g;

    pattern_f = (char *)malloc(sizeof(char) * pattern_len * num_alignments);
    text_f = (char *)malloc(sizeof(char) * text_len * num_alignments);
    for (int i = 0; i < num_alignments; i++) {
        fscanf(fp, "%s", pattern_f + (i * pattern_len));
        fscanf(fp, "%s", text_f + (i * text_len));
    }

    pattern_r = (char *)malloc(sizeof(char) * pattern_len * num_alignments);
    text_r = (char *)malloc(sizeof(char) * text_len * num_alignments);
    for (int j = 0; j < num_alignments; j++) {
        for (int i = 0; i < pattern_len; i++) {
            pattern_r[i + (j * pattern_len)] = pattern_f[((j+1) * pattern_len) - 1 - i];
        }
        for (int i = 0; i < text_len; i++) {
            text_r[i + (j * text_len)] = text_f[((j+1) * text_len) - 1 - i];
        }
    }

    breakpoint_score = (int *)malloc(sizeof(int) * num_alignments);
    for (int i = 0; i < num_alignments; i++) {
        breakpoint_score[i] = INT_MAX;
    }

    // wavefront_components_dimensions_affine()
    int max_score_scope_indel = MAX(penalty_gap_open + penalty_gap_ext, penalty_mismatch) + 1;
    int max_score_scope = MAX(max_score_scope_indel, penalty_mismatch) + 1;

    // wavefront_compute_limits_output()
    int hi = 0;
    int lo = 0;
    int eff_lo = lo - (max_score_scope + 1);
    int eff_hi = hi + (max_score_scope + 1);
    lo = MIN(eff_lo, 0);
    hi = MAX(eff_hi, 0);

    int32_t *offsets, *offsets_g;
    offsets = (int32_t *)malloc(sizeof(int32_t) * wf_length);
    for(int i = 0; i < wf_length; i++) {
        offsets[i] = OFFSET_NULL;
    }

    gpuErrorCheck(hipDeviceReset());
    gpuErrorCheck(hipInit(0));
    gpuErrorCheck(hipSetDevice(0));
    gpuErrorCheck(hipMalloc(&pattern_f_g, sizeof(char) * pattern_len * num_alignments));
    gpuErrorCheck(hipMalloc(&pattern_r_g, sizeof(char) * pattern_len * num_alignments));
    gpuErrorCheck(hipMalloc(&text_f_g, sizeof(char) * text_len * num_alignments));
    gpuErrorCheck(hipMalloc(&text_r_g, sizeof(char) * text_len * num_alignments));
    gpuErrorCheck(hipMalloc(&breakpoint_score_g, sizeof(int) * num_alignments));
    gpuErrorCheck(hipMalloc(&matrix_wf_m_f_g, sizeof(int32_t) * num_wavefronts * wf_length * num_alignments));
    gpuErrorCheck(hipMalloc(&matrix_wf_i_f_g, sizeof(int32_t) * num_wavefronts * wf_length * num_alignments));
    gpuErrorCheck(hipMalloc(&matrix_wf_d_f_g, sizeof(int32_t) * num_wavefronts * wf_length * num_alignments));
    gpuErrorCheck(hipMalloc(&matrix_wf_m_r_g, sizeof(int32_t) * num_wavefronts * wf_length * num_alignments));
    gpuErrorCheck(hipMalloc(&matrix_wf_i_r_g, sizeof(int32_t) * num_wavefronts * wf_length * num_alignments));
    gpuErrorCheck(hipMalloc(&matrix_wf_d_r_g, sizeof(int32_t) * num_wavefronts * wf_length * num_alignments));
    gpuErrorCheck(hipMalloc(&mwavefronts_f_g, sizeof(wf_t) * num_wavefronts * num_alignments));
    gpuErrorCheck(hipMalloc(&iwavefronts_f_g, sizeof(wf_t) * num_wavefronts * num_alignments));
    gpuErrorCheck(hipMalloc(&dwavefronts_f_g, sizeof(wf_t) * num_wavefronts * num_alignments));
    gpuErrorCheck(hipMalloc(&mwavefronts_r_g, sizeof(wf_t) * num_wavefronts * num_alignments));
    gpuErrorCheck(hipMalloc(&iwavefronts_r_g, sizeof(wf_t) * num_wavefronts * num_alignments));
    gpuErrorCheck(hipMalloc(&dwavefronts_r_g, sizeof(wf_t) * num_wavefronts * num_alignments));
    gpuErrorCheck(hipMalloc(&offsets_g, sizeof(int32_t) * wf_length));

    gpuErrorCheck(hipMemcpy(breakpoint_score_g, breakpoint_score, sizeof(int) * num_alignments, hipMemcpyHostToDevice));
    gpuErrorCheck(hipMemcpy(pattern_f_g, pattern_f, sizeof(char) * pattern_len * num_alignments, hipMemcpyHostToDevice));
    gpuErrorCheck(hipMemcpy(pattern_r_g, pattern_r, sizeof(char) * pattern_len * num_alignments, hipMemcpyHostToDevice));
    gpuErrorCheck(hipMemcpy(text_f_g, text_f, sizeof(char) * text_len * num_alignments, hipMemcpyHostToDevice));
    gpuErrorCheck(hipMemcpy(text_r_g, text_r, sizeof(char) * text_len * num_alignments, hipMemcpyHostToDevice));
    gpuErrorCheck(hipMemcpy(offsets_g, offsets, sizeof(int32_t) * wf_length, hipMemcpyHostToDevice));

    dim3 blocksPerGrid(num_alignments, 1, 1);
    dim3 threadsPerBlock(NUM_THREADS, 1, 1);

    std::chrono::high_resolution_clock::time_point start = NOW;
    biWFA<<<blocksPerGrid, threadsPerBlock>>>(pattern_f_g, text_f_g, pattern_r_g, text_r_g, breakpoint_score_g, mwavefronts_f_g,
        iwavefronts_f_g, dwavefronts_f_g, mwavefronts_r_g, iwavefronts_r_g, dwavefronts_r_g, lo, hi, offsets_g,
        pattern_len, text_len, max_score_scope, matrix_wf_m_f_g, matrix_wf_i_f_g, matrix_wf_d_f_g, matrix_wf_m_r_g,
        matrix_wf_i_r_g, matrix_wf_d_r_g);
    gpuErrorCheck(hipDeviceSynchronize());

    std::chrono::high_resolution_clock::time_point end = NOW;
    std::chrono::duration<double> time_temp = (end - start);

    gpuErrorCheck(hipMemcpy(breakpoint_score, breakpoint_score_g, sizeof(int) * num_alignments, hipMemcpyDeviceToHost));

    long double gcups = pattern_len * text_len;
    gcups /= 1E9;
    gcups /= time_temp.count();
    gcups *= num_alignments;
    printf("GPU Time: %lf\n", time_temp.count());
    printf("Estimated GCUPS GPU: : %Lf\n", gcups);

    gpuErrorCheck(hipFree(pattern_f_g));
    gpuErrorCheck(hipFree(pattern_r_g));
    gpuErrorCheck(hipFree(text_f_g));
    gpuErrorCheck(hipFree(text_r_g));
    gpuErrorCheck(hipFree(breakpoint_score_g));
    gpuErrorCheck(hipFree(matrix_wf_m_f_g));
    gpuErrorCheck(hipFree(matrix_wf_i_f_g));
    gpuErrorCheck(hipFree(matrix_wf_d_f_g));
    gpuErrorCheck(hipFree(matrix_wf_m_r_g));
    gpuErrorCheck(hipFree(matrix_wf_i_r_g));
    gpuErrorCheck(hipFree(matrix_wf_d_r_g));
    gpuErrorCheck(hipFree(mwavefronts_f_g));
    gpuErrorCheck(hipFree(iwavefronts_f_g));
    gpuErrorCheck(hipFree(dwavefronts_f_g));
    gpuErrorCheck(hipFree(mwavefronts_r_g));
    gpuErrorCheck(hipFree(iwavefronts_r_g));
    gpuErrorCheck(hipFree(dwavefronts_r_g));
    gpuErrorCheck(hipFree(offsets_g));

    if(s_flag) {
        printf("Scores:\n");
        for (int i = 0; i < num_alignments; i++) {
            printf("%.*s\n%.*s : %d\n", pattern_len, &pattern_f[i * pattern_len], text_len, &text_f[i * text_len], breakpoint_score[i]);
        }
    }

    return 0;
}
