#define num_wavefronts 10
#define wf_length 256

typedef struct _wf_t {
    bool null;
    int lo;
    int hi;
    int32_t offsets[wf_length];
    int wf_elements_init_max;
    int wf_elements_init_min;
} wf_t;

typedef struct _wf_alignment_t {
    char *pattern;
    char *text;
    int num_null_steps;
    int historic_max_hi;
    int historic_min_lo;
} wf_alignment_t;

typedef struct _wf_components_t {
    wf_t mwavefronts[num_wavefronts];
    wf_t dwavefronts[num_wavefronts];
    wf_t iwavefronts[num_wavefronts];
    wf_t wavefront_null;
    wf_alignment_t alignment;
} wf_components_t;
