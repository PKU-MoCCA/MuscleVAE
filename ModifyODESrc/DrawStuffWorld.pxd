# Add by Yulong Zhang
from ModifyODE cimport *
cdef extern from "drawstuff/drawstuffWrapper.h":

    void dsWorldSetter(dWorldID value)
    dWorldID dsWorldGetter()    
    void dsTrackBodyWrapper(dBodyID target, int track_character, int sync_y)
    void dsAssignJointRadius(float x)
    void dsAssignAxisLength(float x)
    void dsCameraLookAtWrapper(float pos_x, float pos_y, float pos_z, float target_x, float target_y, float target_z, float up_x, float up_y, float up_z)
    void dsAssignColor(float red, float green, float blue)
    
    void dsAssignMuscleProperty(float*, double*)
    
    void dsAssignMusclePropertyWithRef(float*, double*, double *)
    
    void dsAssignMuscleHoldAnchorNum(int, int, int*)
    
    void dsAssignMusclePartAndColor(int, int *, float *, double *)
    
    void dsAssignMusclePartAndColorAll(int, int *, float *, double *, double *, double *, double * )
    void dsAssignPauseTime(int value)
    void dsAssignBackground(int x)
    void dsWhetherHingeAxis(int x)
    void dsWhetherLocalAxis(int x)
    void dsDrawWorldinThread()
    void dsDrawWorldinThread()
    
    void dsDrawWorldinThreadYesRender()
    void dsDrawWorldinThreadWithMuscleWithRef()
    void dsDrawWorldinThreadOnlyMuscleWithRef()
    
    void dsKillThread()
    void dsSlowforRender()

    void dsSetWindowWidth(int value)
    void dsSetWindowHeight(int value)
    int dsGetWindowWidth()
    int dsGetWindowHeight()
    void dsGetScreenBuffer(unsigned char * data)

    void dsStartRecordVideo()
    void dsPauseRecordVideo()
    size_t dsGetVideoFrame()
    void dsEndRecordVideo(unsigned char * output_data, size_t num_frame)
