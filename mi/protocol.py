
DEFAULT_PROTOCOL = {
    "fixation_s": 1.0,
    "cue_s": 1.0,
    "imagery_s": 4.0,
    "relax_s": 2.0,
    "jitter_s": 0.5,          # +/- jitter
    "epoch_offset_s": 0.5,    # start feature window after imagery begins
    "epoch_len_s": 3.0,       # length of feature window
    "classes": ["UP", "DOWN", "LEFT", "RIGHT"],
}