def viewer_start(render_object, render_mode, sim_character):
    # rendering muscles and capsules both simulation character and reference character
    if render_mode == 0:
        render_object.start_with_muscle_with_ref(sim_character.muscle_hold_anchor_num, 
                                                 sim_character.anchors_num)
    # only rendering muscles both simulation character and reference character
    elif render_mode == 1:
        render_object.start_only_muscle_with_ref(sim_character.muscle_hold_anchor_num, 
                                                 sim_character.anchors_num)
    else:
        raise NotImplementedError