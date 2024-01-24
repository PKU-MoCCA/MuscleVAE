#include <ode/ode.h>
// #define WIN32 // Add by Zhenhua Song for debug
#ifdef WIN32
#include <vector>
#include <thread>
#include <memory>
#include <mutex>
#include <chrono>
#include <iostream>
#include <GL/glew.h>
#include "drawstuff/drawstuffWrapper.h"

static float color[3] = {1.0f, 1.0f, 0.0f};
static float JointRadius = 0.05f;
static float AxisLength = 0.0f;

dWorldID dsWorld;
static int pauseTime;
static int joint_num;
static std::vector<dJointID> joint_list;
static int draw_axis_num;
static std::vector<dGeomID> draw_axis_list;

static int ds_window_width = 1280;
static int ds_window_height = 720;
static const int ds_max_buffer_length = 1280 * 1280 * 4;
static unsigned char ds_screen_buffer[ds_max_buffer_length];


static double MuscleAnchorRadius = 0.004;
static int muscle_num;
static int anchor_num;
static const int max_muscle_num = 300;
static const int max_anchor_num = 1300;
static int muscle_hold_anchor_num[max_muscle_num];
static int muscle_part_idx[max_muscle_num];
static float muscle_colors_with_input_part_colors[max_muscle_num * 3];
static float muscle_activation[max_muscle_num];
static float muscle_residual_capacity[max_muscle_num];
static float muscle_activated_proportion[max_muscle_num];
static float muscle_resting_proportion[max_muscle_num];
static float muscle_fatigued_proportion[max_muscle_num];
static double anchor_pos[max_anchor_num * 3];

static float ref_anchor_pos[max_anchor_num * 3];
static float residual_capacity_shift_pos[3];
static float activated_proportion_shift_pos[3];
static float resting_proportion_shift_pos[3];
static float fatigued_proportion_shift_pos[3];

// This implementation requires too much memory
// we can export to jpgs for less memory usage

struct dsVideoRecorderCompress
{
};

struct dsVideoRecorder
{
	std::vector<unsigned char *> buffer;
	int enable_flag = 0;
	dsVideoRecorder()
	{
	}
	~dsVideoRecorder()
	{
		clear();
	}

	void clear()
	{
		for (size_t i = 0; i < buffer.size(); i++)
		{
			delete[] buffer[i];
			buffer[i] = nullptr;
		}
		buffer.clear();
	}

	void recordFrame(unsigned char *buf)
	{
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glReadBuffer(GL_FRONT);
		glReadPixels(0, 0, ds_window_width, ds_window_height, GL_BGR_EXT, GL_UNSIGNED_BYTE, buf);
	}

	void appendFrame()
	{
		if (buffer.size() > 1000000)
		{
			return;
		}
		unsigned char *mem = new unsigned char[ds_window_width * ds_window_height * 3];
		recordFrame(mem);
		buffer.push_back(mem);
	}

	int is_enable()
	{
		return enable_flag;
	}

	void set_enable()
	{
		enable_flag = 1;
	}

	void set_disable()
	{
		enable_flag = 0;
	}

	size_t total_frame()
	{
		return buffer.size();
	}
};

static dsVideoRecorder video_recorder;

static GLuint mesh_index;

void dsWorldSetter(dWorldID value)
{
	std::cout << "World Setter Called\n";
	dsWorld = value;
	std::cout << dsWorld << std::endl;
}

dWorldID dsWorldGetter()
{
	return dsWorld;
}

void dsAssignJointRadius(float x)
{
	JointRadius = x;
}

void dsAssignAxisLength(float x)
{
	AxisLength = x;
}

int HINGEAXIS = 0;
void dsWhetherHingeAxis(int x)
{
	HINGEAXIS = x;
}

int LOCALAXIS = 0;
void dsWhetherLocalAxis(int x)
{
	LOCALAXIS = x;
}

// void dsCheckViewpoint(){
//   float pos[3];
//   float rot[3];
//   dsGetViewpoint(pos, rot);
//   std::cout<<"Viewpoint: "<<pos[0]<<" "<<pos[1]<<" "<<pos[2]<<" "<<rot[0]<<" "<<rot[1]<<" "<<rot[2]<<std::endl;
// }

void dsAssignColor(float red, float green, float blue)
{
	color[0] = red;
	color[1] = green;
	color[2] = blue;
}

void dsAssignPauseTime(int value)
{
	pauseTime = value;
}



// dsAssignMuscleHoldAnchorNum(<int>muscle_hold_anchor_num.size ,<int>anchor_num, <int*>muscle_hold_anchor_num.data)
void dsAssignMuscleHoldAnchorNum(int input_muscle_num, int input_anchor_num, int *input_muscle_hold_anchor_num)
{
	muscle_num = input_muscle_num;
	anchor_num = input_anchor_num;
	for (int i = 0; i < muscle_num; ++i)
	{
		muscle_hold_anchor_num[i] = input_muscle_hold_anchor_num[i];
	}
}


void dsAssignMusclePartAndColor(int input_muscle_num, int *input_muscle_part_idx, float *input_part_colors, double *input_residual_capacity_shift_pos)
{
	muscle_num = input_muscle_num;
	for (int i = 0; i < 3; ++i)
	{
		residual_capacity_shift_pos[i] = input_residual_capacity_shift_pos[i];
	}
	for (int i = 0; i < muscle_num; ++i)
	{
		muscle_part_idx[i] = input_muscle_part_idx[i];
	}
	for (int i = 0; i < muscle_num; ++i)
	{
		int idx = input_muscle_part_idx[i];
		muscle_colors_with_input_part_colors[3 * i + 0] = input_part_colors[3 * idx + 0];
		muscle_colors_with_input_part_colors[3 * i + 1] = input_part_colors[3 * idx + 1];
		muscle_colors_with_input_part_colors[3 * i + 2] = input_part_colors[3 * idx + 2];
	}
}


void dsAssignMusclePartAndColorAll(int input_muscle_num, int *input_muscle_part_idx, float *input_part_colors, double *input_residual_capacity_shift_pos, double *input_activated_proportion_shift_pos, double *input_resting_proportion_shift_pos, double *input_fatigued_proportion_shift_pos)
{
	muscle_num = input_muscle_num;
	for (int i = 0; i < 3; ++i)
	{
		residual_capacity_shift_pos[i] = input_residual_capacity_shift_pos[i];
		activated_proportion_shift_pos[i] = input_activated_proportion_shift_pos[i];
		resting_proportion_shift_pos[i] = input_resting_proportion_shift_pos[i];
		fatigued_proportion_shift_pos[i] = input_fatigued_proportion_shift_pos[i];
	}
	for (int i = 0; i < muscle_num; ++i)
	{
		muscle_part_idx[i] = input_muscle_part_idx[i];
	}
	for (int i = 0; i < muscle_num; ++i)
	{
		int idx = input_muscle_part_idx[i];
		muscle_colors_with_input_part_colors[3 * i + 0] = input_part_colors[3 * idx + 0];
		muscle_colors_with_input_part_colors[3 * i + 1] = input_part_colors[3 * idx + 1];
		muscle_colors_with_input_part_colors[3 * i + 2] = input_part_colors[3 * idx + 2];
	}
}


void dsAssignMuscleProperty(float *input_muscle_activation, double *input_anchor_pos)
{
	for (int i = 0; i < muscle_num; ++i)
	{
		muscle_activation[i] = input_muscle_activation[i];
	}
	for (int i = 0; i < anchor_num * 3; ++i)
	{
		anchor_pos[i] = input_anchor_pos[i];
	}
}


void dsAssignMusclePropertyWithRef(float *input_muscle_activation, double *input_anchor_pos, double *input_ref_anchor_pos)
{
	for (int i = 0; i < muscle_num; ++i)
	{
		muscle_activation[i] = input_muscle_activation[i];
	}
	for (int i = 0; i < anchor_num * 3; ++i)
	{
		anchor_pos[i] = input_anchor_pos[i];
	}
	for (int i = 0; i < anchor_num * 3; ++i)
	{
		ref_anchor_pos[i] = input_ref_anchor_pos[i];
	}
}


void dsAssignBackground(int x)
{
	DRAWBACKGROUND = x;
}

void drawStart()
{
	mesh_index = glGenLists(10);

	joint_num = dWorldGetNumBallAndHingeJoints(dsWorld);
	draw_axis_num = 0;
	int type = 0;
	dJointID j = dWorldGetFirstJoint(dsWorld);

	dGeomID g;
	dBodyID b = dWorldGetFirstBody(dsWorld);
	dVector3 v0, v1, v2;

	int meshCnt = 0;
	for (int body_cnt = 0; body_cnt < dsWorld->nb; body_cnt++)
	{
		g = b->geom;
		b = dWorldGetNextBody(b);
		while (g != NULL)
		{
			if (dGeomGetDrawAxisFlag(g))
			{
				draw_axis_list.push_back(g);
				draw_axis_num++;
			}

			type = dGeomGetClass(g);
			if (type == dTriMeshClass)
			{
				int triCnt = dGeomTriMeshGetTriangleCount(g);
				int vCnt = triCnt * 3;
				float *v_pos = new float[triCnt * 9 + 10];
				float *normal = new float[triCnt * 3 + 10];
				for (int i = 0; i < triCnt; i++)
				{
					dGeomTriMeshGetOriginTriangle(g, i, &v0, &v1, &v2);
					for (int j = 0; j < 3; j++)
					{
						v_pos[9 * i + j] = static_cast<float>(v0[j]);
						v_pos[9 * i + 3 + j] = static_cast<float>(v1[j]);
						v_pos[9 * i + 6 + j] = static_cast<float>(v2[j]);
					}
				}
				for (int i = 0; i < triCnt; ++i)
				{
					float u[3], v[3];
					for (int j = 0; j < 3; ++j)
					{
						u[j] = v_pos[9 * i + j] - v_pos[9 * i + 3 + j];
						v[j] = v_pos[9 * i + j] - v_pos[9 * i + 6 + j];
					}
					normal[3 * i] = u[1] * v[2] - u[2] * v[1];
					normal[3 * i + 1] = u[2] * v[0] - u[0] * v[2];
					normal[3 * i + 2] = u[0] * v[1] - u[1] * v[0];

					float len = normal[3 * i] * normal[3 * i] + normal[3 * i + 1] * normal[3 * i + 1] + normal[3 * i + 2] * normal[3 * i + 2];
					if (len <= 0.0f)
					{
						normal[3 * i] = 1;
						normal[3 * i + 1] = 0;
						normal[3 * i + 2] = 0;
					}
					else
					{
						len = 1.0f / (float)sqrt(len);
						normal[3 * i] *= len;
						normal[3 * i + 1] *= len;
						normal[3 * i + 2] *= len;
					}
				}
				glNewList(mesh_index + meshCnt, GL_COMPILE); // compile the first one
				glBegin(GL_TRIANGLES);
				for (int i = 0; i < triCnt; ++i)
				{
					glNormal3fv(normal + 3 * i);
					glVertex3fv(v_pos + 9 * i);
					glVertex3fv(v_pos + 9 * i + 3);
					glVertex3fv(v_pos + 9 * i + 6);
				}
				glEnd();
				glEndList();
				meshCnt++;
				delete[] v_pos;
				delete[] normal;
			}
			g = dGeomGetBodyNext(g);
		}
	}
	// glListBase(mesh_index);
	while (j != NULL)
	{
		type = dJointGetType(j);
		if (type == dJointTypeBall || type == dJointTypeHinge)
		{
			joint_list.push_back(j);
		}
		j = dWorldGetNextJoint(j);
	}
}


void drawStartWithMuscle()
{
	mesh_index = glGenLists(10);

	joint_num = dWorldGetNumBallAndHingeJoints(dsWorld);
	draw_axis_num = 0;
	int type = 0;
	dJointID j = dWorldGetFirstJoint(dsWorld);

	dGeomID g;
	dBodyID b = dWorldGetFirstBody(dsWorld);
	dVector3 v0, v1, v2;

	int meshCnt = 0;
	for (int body_cnt = 0; body_cnt < dsWorld->nb; body_cnt++)
	{
		g = b->geom;
		b = dWorldGetNextBody(b);
		while (g != NULL)
		{
			if (dGeomGetDrawAxisFlag(g))
			{
				draw_axis_list.push_back(g);
				draw_axis_num++;
			}

			type = dGeomGetClass(g);
			if (type == dTriMeshClass)
			{
				int triCnt = dGeomTriMeshGetTriangleCount(g);
				int vCnt = triCnt * 3;
				float *v_pos = new float[triCnt * 9 + 10];
				float *normal = new float[triCnt * 3 + 10];
				for (int i = 0; i < triCnt; i++)
				{
					dGeomTriMeshGetOriginTriangle(g, i, &v0, &v1, &v2);
					for (int j = 0; j < 3; j++)
					{
						v_pos[9 * i + j] = static_cast<float>(v0[j]);
						v_pos[9 * i + 3 + j] = static_cast<float>(v1[j]);
						v_pos[9 * i + 6 + j] = static_cast<float>(v2[j]);
					}
				}
				for (int i = 0; i < triCnt; ++i)
				{
					float u[3], v[3];
					for (int j = 0; j < 3; ++j)
					{
						u[j] = v_pos[9 * i + j] - v_pos[9 * i + 3 + j];
						v[j] = v_pos[9 * i + j] - v_pos[9 * i + 6 + j];
					}
					normal[3 * i] = u[1] * v[2] - u[2] * v[1];
					normal[3 * i + 1] = u[2] * v[0] - u[0] * v[2];
					normal[3 * i + 2] = u[0] * v[1] - u[1] * v[0];

					float len = normal[3 * i] * normal[3 * i] + normal[3 * i + 1] * normal[3 * i + 1] + normal[3 * i + 2] * normal[3 * i + 2];
					if (len <= 0.0f)
					{
						normal[3 * i] = 1;
						normal[3 * i + 1] = 0;
						normal[3 * i + 2] = 0;
					}
					else
					{
						len = 1.0f / (float)sqrt(len);
						normal[3 * i] *= len;
						normal[3 * i + 1] *= len;
						normal[3 * i + 2] *= len;
					}
				}
				glNewList(mesh_index + meshCnt, GL_COMPILE); // compile the first one
				glBegin(GL_TRIANGLES);
				for (int i = 0; i < triCnt; ++i)
				{
					glNormal3fv(normal + 3 * i);
					glVertex3fv(v_pos + 9 * i);
					glVertex3fv(v_pos + 9 * i + 3);
					glVertex3fv(v_pos + 9 * i + 6);
				}
				glEnd();
				glEndList();
				meshCnt++;
				delete[] v_pos;
				delete[] normal;
			}
			g = dGeomGetBodyNext(g);
		}
	}
	// glListBase(mesh_index);
	while (j != NULL)
	{
		type = dJointGetType(j);
		if (type == dJointTypeBall || type == dJointTypeHinge)
		{
			joint_list.push_back(j);
		}
		j = dWorldGetNextJoint(j);
	}
}


void drawStartYesRender()
{
	mesh_index = glGenLists(10);

	joint_num = dWorldGetNumBallAndHingeJoints(dsWorld);
	draw_axis_num = 0;
	int type = 0;
	dJointID j = dWorldGetFirstJoint(dsWorld);

	dGeomID g;
	dBodyID b = dWorldGetFirstBody(dsWorld);
	dVector3 v0, v1, v2;

	int meshCnt = 0;
	int tmp_whether_render = 1;
	// std::cout << "laal" << std::endl; // DEBUG
	for (int body_cnt = 0; body_cnt < dsWorld->nb; body_cnt++)
	{
		// std::cout << "laal" << std::endl; //DEBUG
		g = b->geom;
		tmp_whether_render = b->whether_render;
		b = dWorldGetNextBody(b);
		while (g != NULL && tmp_whether_render)
		{
			if (dGeomGetDrawAxisFlag(g))
			{
				draw_axis_list.push_back(g);
				draw_axis_num++;
			}

			type = dGeomGetClass(g);
			if (type == dTriMeshClass)
			{
				int triCnt = dGeomTriMeshGetTriangleCount(g);
				int vCnt = triCnt * 3;
				float *v_pos = new float[triCnt * 9 + 10];
				float *normal = new float[triCnt * 3 + 10];
				for (int i = 0; i < triCnt; i++)
				{
					dGeomTriMeshGetOriginTriangle(g, i, &v0, &v1, &v2);
					for (int j = 0; j < 3; j++)
					{
						v_pos[9 * i + j] = static_cast<float>(v0[j]);
						v_pos[9 * i + 3 + j] = static_cast<float>(v1[j]);
						v_pos[9 * i + 6 + j] = static_cast<float>(v2[j]);
					}
				}
				for (int i = 0; i < triCnt; ++i)
				{
					float u[3], v[3];
					for (int j = 0; j < 3; ++j)
					{
						u[j] = v_pos[9 * i + j] - v_pos[9 * i + 3 + j];
						v[j] = v_pos[9 * i + j] - v_pos[9 * i + 6 + j];
					}
					normal[3 * i] = u[1] * v[2] - u[2] * v[1];
					normal[3 * i + 1] = u[2] * v[0] - u[0] * v[2];
					normal[3 * i + 2] = u[0] * v[1] - u[1] * v[0];

					float len = normal[3 * i] * normal[3 * i] + normal[3 * i + 1] * normal[3 * i + 1] + normal[3 * i + 2] * normal[3 * i + 2];
					if (len <= 0.0f)
					{
						normal[3 * i] = 1;
						normal[3 * i + 1] = 0;
						normal[3 * i + 2] = 0;
					}
					else
					{
						len = 1.0f / (float)sqrt(len);
						normal[3 * i] *= len;
						normal[3 * i + 1] *= len;
						normal[3 * i + 2] *= len;
					}
				}
				glNewList(mesh_index + meshCnt, GL_COMPILE); // compile the first one
				glBegin(GL_TRIANGLES);
				for (int i = 0; i < triCnt; ++i)
				{
					glNormal3fv(normal + 3 * i);
					glVertex3fv(v_pos + 9 * i);
					glVertex3fv(v_pos + 9 * i + 3);
					glVertex3fv(v_pos + 9 * i + 6);
				}
				glEnd();
				glEndList();
				meshCnt++;
				delete[] v_pos;
				delete[] normal;
			}
			g = dGeomGetBodyNext(g);
		}
	}
	// glListBase(mesh_index);
	while (j != NULL)
	{
		type = dJointGetType(j);
		if (type == dJointTypeBall || type == dJointTypeHinge)
		{
			joint_list.push_back(j);
		}
		j = dWorldGetNextJoint(j);
	}
}

// Add by Zhenhua Song
static void ds_output_image_to_file(const char *filename)
{
	FILE *outputFile = fopen(filename, "w");
	short header[] = {0, 2, 0, 0, 0, 0, (short)ds_window_width, (short)ds_window_height, 24};

	fwrite(&header, sizeof(header), 1, outputFile);
	fwrite(ds_screen_buffer, ds_window_height * ds_window_width * 3, 1, outputFile);
	fclose(outputFile);

	std::cout << "Finish writing to file.\n"
			  << std::endl;
}

void dsDrawStep(int pause)
{
	// dsCheckViewpoint();
	static float pos[3];
	static float rot[12];
	static dxPosR posr_origin;
	static dxPosR *posr = &posr_origin;
	dGeomID g;
	dBodyID b = dWorldGetFirstBody(dsWorld);
	dJointID j = dWorldGetFirstJoint(dsWorld);
	dReal l, r;
	int type;

	int meshCnt = 0;
	static dVector3 result1, result2, haxis_result;
	static float axis_norm, pos1[3], pos2[3];

	dsSetColor(color[0], color[1], color[2]);
	for (int body_cnt = 0; body_cnt < dsWorld->nb; body_cnt++)
	{
		g = b->geom;
		b = dWorldGetNextBody(b);
		while (g != NULL)
		{
			// This part is add by Zhenhua Song
			int use_default_color = dGeomIsRenderInDefaultColor(g);
			// for debug..
			// fout << "use_default_color: " << use_default_color << std::endl;
			if (use_default_color)
			{
				dsSetColor(color[0], color[1], color[2]);
			}
			else
			{
				double user_color[3];
				dGeomRenderGetUserColor(g, user_color);
				// dsAssignColor()
				dsSetColor(static_cast<float>(user_color[0]), static_cast<float>(user_color[1]), static_cast<float>(user_color[2]));
				// fout << "user color: " << user_color[0] << " " << user_color[1] << " " << user_color[2] << std::endl;
			}
			// end Add by Zhenhua Song.
			dGeomCopyPosRot4Render(g, posr);
			for (int i = 0; i < 3; i++)
				pos[i] = static_cast<float>(posr->pos[i]);
			for (int i = 0; i < 12; i++)
				rot[i] = static_cast<float>(posr->R[i]);
			type = dGeomGetClass(g);
			switch (type)
			{
			case dSphereClass:
				dsDrawSphere(pos, rot, static_cast<float>(dGeomSphereGetRadius(g)));
				break;
			case dBoxClass:
				float sides[3];
				dVector3 box_size;
				dGeomBoxGetLengths(g, box_size);
				sides[0] = static_cast<float>(box_size[0]);
				sides[1] = static_cast<float>(box_size[1]);
				sides[2] = static_cast<float>(box_size[2]);
				dsDrawBox(pos, rot, sides);
				break;
			case dCapsuleClass:
				dGeomCapsuleGetParams(g, &r, &l);
				dsDrawCapsule(pos, rot, static_cast<float>(l), static_cast<float>(r));
				break;
			case dCylinderClass:
				dGeomCylinderGetParams(g, &r, &l);
				dsDrawCylinder(pos, rot, static_cast<float>(l), static_cast<float>(r));
				break;
			case dTriMeshClass:
				dsDrawTriMesh1(pos, rot, mesh_index + meshCnt);
				meshCnt++;
				break;
			}
			g = dGeomGetBodyNext(g);
		}
	}
	if (HINGEAXIS)
	{
		for (int k = 0; k < 12; k++)
			rot[k] = 0.0f;
		rot[0] = 1.0f;
		rot[5] = 1.0f;
		rot[10] = 1.0f;
		for (int i = 0; i < joint_num; i++)
		{
			type = dJointGetType(joint_list[i]);
			switch (type)
			{
			case dJointTypeBall:
				dJointGetBallAnchor(joint_list[i], result1);
				dJointGetBallAnchor2(joint_list[i], result2);
				// draw anchor 1 in red
				pos[0] = static_cast<float>(result1[0]);
				pos[1] = static_cast<float>(result1[1]);
				pos[2] = static_cast<float>(result1[2]);
				dsSetColor(1.0f, 0.0f, 0.0f);
				dsDrawSphere(pos, rot, JointRadius);
				// draw anchor 2 in blue
				pos[0] = static_cast<float>(result2[0]);
				pos[1] = static_cast<float>(result2[1]);
				pos[2] = static_cast<float>(result2[2]);
				dsSetColor(0.0f, 0.0f, 1.0f);
				dsDrawSphere(pos, rot, JointRadius);
				break;
			case dJointTypeHinge:
				dJointGetHingeAnchor(joint_list[i], result1);
				dJointGetHingeAnchor2(joint_list[i], result2);
				pos[0] = static_cast<float>(result1[0]);
				pos[1] = static_cast<float>(result1[1]);
				pos[2] = static_cast<float>(result1[2]);
				// draw anchor 1 in red
				pos[0] = static_cast<float>(result1[0]);
				pos[1] = static_cast<float>(result1[1]);
				pos[2] = static_cast<float>(result1[2]);
				dsSetColor(1.0f, 0.0f, 0.0f);
				dsDrawSphere(pos, rot, JointRadius);
				// draw anchor 2 in blue
				pos[0] = static_cast<float>(result2[0]);
				pos[1] = static_cast<float>(result2[1]);
				pos[2] = static_cast<float>(result2[2]);
				dsSetColor(0.0f, 0.0f, 1.0f);
				dsDrawSphere(pos, rot, JointRadius);
				dJointGetHingeAxis1(joint_list[i], haxis_result);
				axis_norm = static_cast<float>(AxisLength / sqrt(haxis_result[0] * haxis_result[0] + haxis_result[1] * haxis_result[1] + haxis_result[2] * haxis_result[2]));
				pos1[0] = static_cast<float>(result1[0]);
				pos2[0] = static_cast<float>(pos1[0] + axis_norm * haxis_result[0]);
				pos1[1] = static_cast<float>(result1[1]);
				pos2[1] = static_cast<float>(pos1[1] + axis_norm * haxis_result[1]);
				pos1[2] = static_cast<float>(result1[2]);
				pos2[2] = static_cast<float>(pos1[2] + axis_norm * haxis_result[2]);
				dsSetColor(1.0f, 0.0f, 0.0f);
				dsDrawLine(pos1, pos2);
				dJointGetHingeAxis2(joint_list[i], haxis_result);
				axis_norm = static_cast<float>(AxisLength / sqrt(haxis_result[0] * haxis_result[0] + haxis_result[1] * haxis_result[1] + haxis_result[2] * haxis_result[2]));
				pos1[0] = static_cast<float>(result2[0]);
				pos2[0] = static_cast<float>(pos1[0] + axis_norm * haxis_result[0]);
				pos1[1] = static_cast<float>(result2[1]);
				pos2[1] = static_cast<float>(pos1[1] + axis_norm * haxis_result[1]);
				pos1[2] = static_cast<float>(result2[2]);
				pos2[2] = static_cast<float>(pos1[2] + axis_norm * haxis_result[2]);
				dsSetColor(0.0f, 0.0f, 1.0f);
				dsDrawLine(pos1, pos2);
				break;
			}
		}
	}

	if (LOCALAXIS)
	{
		for (int i = 0; i < draw_axis_num; i++)
		{
			dGeomCopyPosRot4Render(draw_axis_list[i], posr);
			for (int k = 0; k < 3; k++)
				pos[k] = static_cast<float>(posr->pos[k]);
			for (int k = 0; k < 12; k++)
				rot[k] = static_cast<float>(posr->R[k]);

			dsDrawLocalAxis(pos, rot, AxisLength);
		}
	}

	if (video_recorder.is_enable())
	{
		video_recorder.appendFrame();
	}
}


void dsDrawStepWithMuscleWithRef(int pause)
{
	// dsCheckViewpoint();
	static float pos[3];
	static float rot[12];
	static dxPosR posr_origin;
	static dxPosR *posr = &posr_origin;
	dGeomID g;
	dBodyID b = dWorldGetFirstBody(dsWorld);
	dJointID j = dWorldGetFirstJoint(dsWorld);
	dReal l, r;
	int type;

	int meshCnt = 0;
	static dVector3 result1, result2, haxis_result;
	static float axis_norm, pos1[3], pos2[3];

	dsSetColor(color[0], color[1], color[2]);
	for (int body_cnt = 0; body_cnt < dsWorld->nb; body_cnt++)
	{
		g = b->geom;
		b = dWorldGetNextBody(b);
		while (g != NULL)
		{
			// This part is add by Zhenhua Song
			int use_default_color = dGeomIsRenderInDefaultColor(g);
			// for debug..
			// fout << "use_default_color: " << use_default_color << std::endl;
			if (use_default_color)
			{
				dsSetColor(color[0], color[1], color[2]);
			}
			else
			{
				double user_color[3];
				dGeomRenderGetUserColor(g, user_color);
				// dsAssignColor()
				dsSetColor(static_cast<float>(user_color[0]), static_cast<float>(user_color[1]), static_cast<float>(user_color[2]));
				// fout << "user color: " << user_color[0] << " " << user_color[1] << " " << user_color[2] << std::endl;
			}
			// end Add by Zhenhua Song.
			dGeomCopyPosRot4Render(g, posr);
			for (int i = 0; i < 3; i++)
				pos[i] = static_cast<float>(posr->pos[i]);
			for (int i = 0; i < 12; i++)
				rot[i] = static_cast<float>(posr->R[i]);
			type = dGeomGetClass(g);
			switch (type)
			{
			case dSphereClass:
				dsDrawSphere(pos, rot, static_cast<float>(dGeomSphereGetRadius(g)));
				break;
			case dBoxClass:
				float sides[3];
				dVector3 box_size;
				dGeomBoxGetLengths(g, box_size);
				sides[0] = static_cast<float>(box_size[0]);
				sides[1] = static_cast<float>(box_size[1]);
				sides[2] = static_cast<float>(box_size[2]);
				dsDrawBox(pos, rot, sides);
				break;
			case dCapsuleClass:
				dGeomCapsuleGetParams(g, &r, &l);
				dsDrawCapsule(pos, rot, static_cast<float>(l), static_cast<float>(r));
				break;
			case dCylinderClass:
				dGeomCylinderGetParams(g, &r, &l);
				dsDrawCylinder(pos, rot, static_cast<float>(l), static_cast<float>(r));
				break;
			case dTriMeshClass:
				dsDrawTriMesh1(pos, rot, mesh_index + meshCnt);
				meshCnt++;
				break;
			}
			g = dGeomGetBodyNext(g);
		}
	}
	// Draw muscles
	
	for (int k = 0; k < 12; k++)
		rot[k] = 0.0f;
	rot[0] = 1.0f;
	rot[5] = 1.0f;
	rot[10] = 1.0f;
	int till_now_anchor_num = 0;
	for (int muscle_idx = 0; muscle_idx < muscle_num; ++muscle_idx)
	{
		int tmp_anchor_num = muscle_hold_anchor_num[muscle_idx];
		for (int tmp_anchor_idx = 0; tmp_anchor_idx < tmp_anchor_num - 1; ++tmp_anchor_idx)
		{
			// Draw muscle line
			float tmp_muscle_color = muscle_activation[muscle_idx];
			dsSetColor(1.0f, 1.0f-tmp_muscle_color, 1.0f-tmp_muscle_color);
			pos1[0] = anchor_pos[3 * (tmp_anchor_idx + till_now_anchor_num) + 0];
			pos1[1] = anchor_pos[3 * (tmp_anchor_idx + till_now_anchor_num) + 1];
			pos1[2] = anchor_pos[3 * (tmp_anchor_idx + till_now_anchor_num) + 2];
			pos2[0] = anchor_pos[3 * (tmp_anchor_idx + till_now_anchor_num + 1) + 0];
			pos2[1] = anchor_pos[3 * (tmp_anchor_idx + till_now_anchor_num + 1) + 1];
			pos2[2] = anchor_pos[3 * (tmp_anchor_idx + till_now_anchor_num + 1) + 2];
			float tmp_width = 1.0f + 3.0f * tmp_muscle_color;
			dsDrawLineWithWidth(pos1, pos2, tmp_width);
			// Draw anchor point
			dsSetColor(1.0f, 1.0f, 0.0f);
			dsDrawSphere(pos1, rot, MuscleAnchorRadius);
		}
		// Draw anchor point
		dsSetColor(1.0f, 1.0f, 0.0f);
		dsDrawSphere(pos2, rot, MuscleAnchorRadius);
		till_now_anchor_num += tmp_anchor_num;
	}

	// Draw Ref muscles for Check and Debug
	// Anchors are Green
	till_now_anchor_num = 0;
	for (int muscle_idx = 0; muscle_idx < muscle_num; ++muscle_idx)
	{
		int tmp_anchor_num = muscle_hold_anchor_num[muscle_idx];
		for (int tmp_anchor_idx = 0; tmp_anchor_idx < tmp_anchor_num - 1; ++tmp_anchor_idx)
		{
			// Draw muscle line
			// float tmp_muscle_color = muscle_activation[muscle_idx];
			dsSetColor(1.0f, 1.0f, 1.0f);
			pos1[0] = ref_anchor_pos[3 * (tmp_anchor_idx + till_now_anchor_num) + 0];
			pos1[1] = ref_anchor_pos[3 * (tmp_anchor_idx + till_now_anchor_num) + 1];
			pos1[2] = ref_anchor_pos[3 * (tmp_anchor_idx + till_now_anchor_num) + 2];
			pos2[0] = ref_anchor_pos[3 * (tmp_anchor_idx + till_now_anchor_num + 1) + 0];
			pos2[1] = ref_anchor_pos[3 * (tmp_anchor_idx + till_now_anchor_num + 1) + 1];
			pos2[2] = ref_anchor_pos[3 * (tmp_anchor_idx + till_now_anchor_num + 1) + 2];
			dsDrawLineWithWidth(pos1, pos2, 1.0f);
			// Draw anchor point
			dsSetColor(0.0f, 1.0f, 0.0f);
			dsDrawSphere(pos1, rot, MuscleAnchorRadius);
		}
		// Draw anchor point
		dsSetColor(0.0f, 1.0f, 0.0f);
		dsDrawSphere(pos2, rot, MuscleAnchorRadius);
		till_now_anchor_num += tmp_anchor_num;
	}


	if (HINGEAXIS)
	{
		for (int k = 0; k < 12; k++)
			rot[k] = 0.0f;
		rot[0] = 1.0f;
		rot[5] = 1.0f;
		rot[10] = 1.0f;
		for (int i = 0; i < joint_num; i++)
		{
			type = dJointGetType(joint_list[i]);
			switch (type)
			{
			case dJointTypeBall:
				dJointGetBallAnchor(joint_list[i], result1);
				dJointGetBallAnchor2(joint_list[i], result2);
				// draw anchor 1 in red
				pos[0] = static_cast<float>(result1[0]);
				pos[1] = static_cast<float>(result1[1]);
				pos[2] = static_cast<float>(result1[2]);
				dsSetColor(1.0f, 0.0f, 0.0f);
				dsDrawSphere(pos, rot, JointRadius);
				// draw anchor 2 in blue
				pos[0] = static_cast<float>(result2[0]);
				pos[1] = static_cast<float>(result2[1]);
				pos[2] = static_cast<float>(result2[2]);
				dsSetColor(0.0f, 0.0f, 1.0f);
				dsDrawSphere(pos, rot, JointRadius);
				break;
			case dJointTypeHinge:
				dJointGetHingeAnchor(joint_list[i], result1);
				dJointGetHingeAnchor2(joint_list[i], result2);
				pos[0] = static_cast<float>(result1[0]);
				pos[1] = static_cast<float>(result1[1]);
				pos[2] = static_cast<float>(result1[2]);
				// draw anchor 1 in red
				pos[0] = static_cast<float>(result1[0]);
				pos[1] = static_cast<float>(result1[1]);
				pos[2] = static_cast<float>(result1[2]);
				dsSetColor(1.0f, 0.0f, 0.0f);
				dsDrawSphere(pos, rot, JointRadius);
				// draw anchor 2 in blue
				pos[0] = static_cast<float>(result2[0]);
				pos[1] = static_cast<float>(result2[1]);
				pos[2] = static_cast<float>(result2[2]);
				dsSetColor(0.0f, 0.0f, 1.0f);
				dsDrawSphere(pos, rot, JointRadius);
				dJointGetHingeAxis1(joint_list[i], haxis_result);
				axis_norm = static_cast<float>(AxisLength / sqrt(haxis_result[0] * haxis_result[0] + haxis_result[1] * haxis_result[1] + haxis_result[2] * haxis_result[2]));
				pos1[0] = static_cast<float>(result1[0]);
				pos2[0] = static_cast<float>(pos1[0] + axis_norm * haxis_result[0]);
				pos1[1] = static_cast<float>(result1[1]);
				pos2[1] = static_cast<float>(pos1[1] + axis_norm * haxis_result[1]);
				pos1[2] = static_cast<float>(result1[2]);
				pos2[2] = static_cast<float>(pos1[2] + axis_norm * haxis_result[2]);
				dsSetColor(1.0f, 0.0f, 0.0f);
				dsDrawLine(pos1, pos2);
				dJointGetHingeAxis2(joint_list[i], haxis_result);
				axis_norm = static_cast<float>(AxisLength / sqrt(haxis_result[0] * haxis_result[0] + haxis_result[1] * haxis_result[1] + haxis_result[2] * haxis_result[2]));
				pos1[0] = static_cast<float>(result2[0]);
				pos2[0] = static_cast<float>(pos1[0] + axis_norm * haxis_result[0]);
				pos1[1] = static_cast<float>(result2[1]);
				pos2[1] = static_cast<float>(pos1[1] + axis_norm * haxis_result[1]);
				pos1[2] = static_cast<float>(result2[2]);
				pos2[2] = static_cast<float>(pos1[2] + axis_norm * haxis_result[2]);
				dsSetColor(0.0f, 0.0f, 1.0f);
				dsDrawLine(pos1, pos2);
				break;
			}
		}
	}

	if (LOCALAXIS)
	{
		for (int i = 0; i < draw_axis_num; i++)
		{
			dGeomCopyPosRot4Render(draw_axis_list[i], posr);
			for (int k = 0; k < 3; k++)
				pos[k] = static_cast<float>(posr->pos[k]);
			for (int k = 0; k < 12; k++)
				rot[k] = static_cast<float>(posr->R[k]);

			dsDrawLocalAxis(pos, rot, AxisLength);
		}
	}

	if (video_recorder.is_enable())
	{
		video_recorder.appendFrame();
	}
}

void dsDrawStepOnlyMuscleWithRef(int pause)
{
	// dsCheckViewpoint();
	static float pos[3];
	static float rot[12];
	static dxPosR posr_origin;
	static dxPosR *posr = &posr_origin;
	dGeomID g;
	dBodyID b = dWorldGetFirstBody(dsWorld);
	dJointID j = dWorldGetFirstJoint(dsWorld);
	dReal l, r;
	int type;

	int meshCnt = 0;
	static dVector3 result1, result2, haxis_result;
	static float axis_norm, pos1[3], pos2[3];

	dsSetColor(color[0], color[1], color[2]);
	for (int body_cnt = 0; body_cnt < dsWorld->nb; body_cnt++)
	{
		g = b->geom;
		b = dWorldGetNextBody(b);
		while (g != NULL)
		{
			// This part is add by Zhenhua Song
			int use_default_color = dGeomIsRenderInDefaultColor(g);
			if (use_default_color)
			{
				dsSetColor(color[0], color[1], color[2]);
			}
			else
			{
				double user_color[3];
				dGeomRenderGetUserColor(g, user_color);
				// dsAssignColor()
				dsSetColor(static_cast<float>(user_color[0]), static_cast<float>(user_color[1]), static_cast<float>(user_color[2]));
			}
			// end Add by Zhenhua Song.
			dGeomCopyPosRot4Render(g, posr);
			for (int i = 0; i < 3; i++)
				pos[i] = static_cast<float>(posr->pos[i]);
			for (int i = 0; i < 12; i++)
				rot[i] = static_cast<float>(posr->R[i]);
			type = dGeomGetClass(g);
			
			g = dGeomGetBodyNext(g);
		}
	}
	// Draw muscles
	
	for (int k = 0; k < 12; k++)
		rot[k] = 0.0f;
	rot[0] = 1.0f;
	rot[5] = 1.0f;
	rot[10] = 1.0f;
	int till_now_anchor_num = 0;
	for (int muscle_idx = 0; muscle_idx < muscle_num; ++muscle_idx)
	{
		int tmp_anchor_num = muscle_hold_anchor_num[muscle_idx];
		for (int tmp_anchor_idx = 0; tmp_anchor_idx < tmp_anchor_num - 1; ++tmp_anchor_idx)
		{
			// Draw muscle line
			float tmp_muscle_color = muscle_activation[muscle_idx];
			dsSetColor(1.0f, 1.0f-tmp_muscle_color, 1.0f-tmp_muscle_color);
			pos1[0] = anchor_pos[3 * (tmp_anchor_idx + till_now_anchor_num) + 0];
			pos1[1] = anchor_pos[3 * (tmp_anchor_idx + till_now_anchor_num) + 1];
			pos1[2] = anchor_pos[3 * (tmp_anchor_idx + till_now_anchor_num) + 2];
			pos2[0] = anchor_pos[3 * (tmp_anchor_idx + till_now_anchor_num + 1) + 0];
			pos2[1] = anchor_pos[3 * (tmp_anchor_idx + till_now_anchor_num + 1) + 1];
			pos2[2] = anchor_pos[3 * (tmp_anchor_idx + till_now_anchor_num + 1) + 2];
			float tmp_width = 1.0f + 3.0f * tmp_muscle_color;
			dsDrawLineWithWidth(pos1, pos2, tmp_width);
			// Draw anchor point
			dsSetColor(1.0f, 1.0f, 0.0f);
			dsDrawSphere(pos1, rot, MuscleAnchorRadius);
		}
		// Draw anchor point
		dsSetColor(1.0f, 1.0f, 0.0f);
		dsDrawSphere(pos2, rot, MuscleAnchorRadius);
		till_now_anchor_num += tmp_anchor_num;
	}

	// Draw Ref muscles for Check and Debug
	// Anchors are Green
	till_now_anchor_num = 0;
	for (int muscle_idx = 0; muscle_idx < muscle_num; ++muscle_idx)
	{
		int tmp_anchor_num = muscle_hold_anchor_num[muscle_idx];
		for (int tmp_anchor_idx = 0; tmp_anchor_idx < tmp_anchor_num - 1; ++tmp_anchor_idx)
		{
			// Draw muscle line
			// float tmp_muscle_color = muscle_activation[muscle_idx];
			dsSetColor(1.0f, 1.0f, 1.0f);
			pos1[0] = ref_anchor_pos[3 * (tmp_anchor_idx + till_now_anchor_num) + 0];
			pos1[1] = ref_anchor_pos[3 * (tmp_anchor_idx + till_now_anchor_num) + 1];
			pos1[2] = ref_anchor_pos[3 * (tmp_anchor_idx + till_now_anchor_num) + 2];
			pos2[0] = ref_anchor_pos[3 * (tmp_anchor_idx + till_now_anchor_num + 1) + 0];
			pos2[1] = ref_anchor_pos[3 * (tmp_anchor_idx + till_now_anchor_num + 1) + 1];
			pos2[2] = ref_anchor_pos[3 * (tmp_anchor_idx + till_now_anchor_num + 1) + 2];
			dsDrawLineWithWidth(pos1, pos2, 1.0f);
			// Draw anchor point
			dsSetColor(0.0f, 1.0f, 0.0f);
			dsDrawSphere(pos1, rot, MuscleAnchorRadius);
		}
		// Draw anchor point
		dsSetColor(0.0f, 1.0f, 0.0f);
		dsDrawSphere(pos2, rot, MuscleAnchorRadius);
		till_now_anchor_num += tmp_anchor_num;
	}

	if (HINGEAXIS)
	{
		for (int k = 0; k < 12; k++)
			rot[k] = 0.0f;
		rot[0] = 1.0f;
		rot[5] = 1.0f;
		rot[10] = 1.0f;
		for (int i = 0; i < joint_num; i++)
		{
			type = dJointGetType(joint_list[i]);
			switch (type)
			{
			case dJointTypeBall:
				dJointGetBallAnchor(joint_list[i], result1);
				dJointGetBallAnchor2(joint_list[i], result2);
				// draw anchor 1 in red
				pos[0] = static_cast<float>(result1[0]);
				pos[1] = static_cast<float>(result1[1]);
				pos[2] = static_cast<float>(result1[2]);
				dsSetColor(1.0f, 0.0f, 0.0f);
				dsDrawSphere(pos, rot, JointRadius);
				// draw anchor 2 in blue
				pos[0] = static_cast<float>(result2[0]);
				pos[1] = static_cast<float>(result2[1]);
				pos[2] = static_cast<float>(result2[2]);
				dsSetColor(0.0f, 0.0f, 1.0f);
				dsDrawSphere(pos, rot, JointRadius);
				break;
			case dJointTypeHinge:
				dJointGetHingeAnchor(joint_list[i], result1);
				dJointGetHingeAnchor2(joint_list[i], result2);
				pos[0] = static_cast<float>(result1[0]);
				pos[1] = static_cast<float>(result1[1]);
				pos[2] = static_cast<float>(result1[2]);
				// draw anchor 1 in red
				pos[0] = static_cast<float>(result1[0]);
				pos[1] = static_cast<float>(result1[1]);
				pos[2] = static_cast<float>(result1[2]);
				dsSetColor(1.0f, 0.0f, 0.0f);
				dsDrawSphere(pos, rot, JointRadius);
				// draw anchor 2 in blue
				pos[0] = static_cast<float>(result2[0]);
				pos[1] = static_cast<float>(result2[1]);
				pos[2] = static_cast<float>(result2[2]);
				dsSetColor(0.0f, 0.0f, 1.0f);
				dsDrawSphere(pos, rot, JointRadius);
				dJointGetHingeAxis1(joint_list[i], haxis_result);
				axis_norm = static_cast<float>(AxisLength / sqrt(haxis_result[0] * haxis_result[0] + haxis_result[1] * haxis_result[1] + haxis_result[2] * haxis_result[2]));
				pos1[0] = static_cast<float>(result1[0]);
				pos2[0] = static_cast<float>(pos1[0] + axis_norm * haxis_result[0]);
				pos1[1] = static_cast<float>(result1[1]);
				pos2[1] = static_cast<float>(pos1[1] + axis_norm * haxis_result[1]);
				pos1[2] = static_cast<float>(result1[2]);
				pos2[2] = static_cast<float>(pos1[2] + axis_norm * haxis_result[2]);
				dsSetColor(1.0f, 0.0f, 0.0f);
				dsDrawLine(pos1, pos2);
				dJointGetHingeAxis2(joint_list[i], haxis_result);
				axis_norm = static_cast<float>(AxisLength / sqrt(haxis_result[0] * haxis_result[0] + haxis_result[1] * haxis_result[1] + haxis_result[2] * haxis_result[2]));
				pos1[0] = static_cast<float>(result2[0]);
				pos2[0] = static_cast<float>(pos1[0] + axis_norm * haxis_result[0]);
				pos1[1] = static_cast<float>(result2[1]);
				pos2[1] = static_cast<float>(pos1[1] + axis_norm * haxis_result[1]);
				pos1[2] = static_cast<float>(result2[2]);
				pos2[2] = static_cast<float>(pos1[2] + axis_norm * haxis_result[2]);
				dsSetColor(0.0f, 0.0f, 1.0f);
				dsDrawLine(pos1, pos2);
				break;
			}
		}
	}

	if (LOCALAXIS)
	{
		for (int i = 0; i < draw_axis_num; i++)
		{
			dGeomCopyPosRot4Render(draw_axis_list[i], posr);
			for (int k = 0; k < 3; k++)
				pos[k] = static_cast<float>(posr->pos[k]);
			for (int k = 0; k < 12; k++)
				rot[k] = static_cast<float>(posr->R[k]);

			dsDrawLocalAxis(pos, rot, AxisLength);
		}
	}

	if (video_recorder.is_enable())
	{
		video_recorder.appendFrame();
	}
}


void dsDrawStepYesRender(int pause)
{
	// dsCheckViewpoint();
	static float pos[3];
	static float rot[12];
	static dxPosR posr_origin;
	static dxPosR *posr = &posr_origin;
	dGeomID g;
	dBodyID b = dWorldGetFirstBody(dsWorld);
	dJointID j = dWorldGetFirstJoint(dsWorld);
	dReal l, r;
	int type;
	// std::cout << "laal" << std::endl; //DEBUG
	int meshCnt = 0;
	static dVector3 result1, result2, haxis_result;
	static float axis_norm, pos1[3], pos2[3];

	int tmp_whether_render = 1;
	dsSetColor(color[0], color[1], color[2]);
	for (int body_cnt = 0; body_cnt < dsWorld->nb; body_cnt++)
	{
		g = b->geom;
		tmp_whether_render = b->whether_render;
		b = dWorldGetNextBody(b);
		while (g != NULL && tmp_whether_render)
		{
			// This part is add by Zhenhua Song
			int use_default_color = dGeomIsRenderInDefaultColor(g);
			// for debug..
			// fout << "use_default_color: " << use_default_color << std::endl;
			if (use_default_color)
			{
				dsSetColor(color[0], color[1], color[2]);
			}
			else
			{
				double user_color[3];
				dGeomRenderGetUserColor(g, user_color);
				// dsAssignColor()
				dsSetColor(static_cast<float>(user_color[0]), static_cast<float>(user_color[1]), static_cast<float>(user_color[2]));
				// fout << "user color: " << user_color[0] << " " << user_color[1] << " " << user_color[2] << std::endl;
			}
			// end Add by Zhenhua Song.
			dGeomCopyPosRot4Render(g, posr);
			for (int i = 0; i < 3; i++)
				pos[i] = static_cast<float>(posr->pos[i]);
			for (int i = 0; i < 12; i++)
				rot[i] = static_cast<float>(posr->R[i]);
			type = dGeomGetClass(g);
			switch (type)
			{
			case dSphereClass:
				dsDrawSphere(pos, rot, static_cast<float>(dGeomSphereGetRadius(g)));
				break;
			case dBoxClass:
				float sides[3];
				dVector3 box_size;
				dGeomBoxGetLengths(g, box_size);
				sides[0] = static_cast<float>(box_size[0]);
				sides[1] = static_cast<float>(box_size[1]);
				sides[2] = static_cast<float>(box_size[2]);
				dsDrawBox(pos, rot, sides);
				break;
			case dCapsuleClass:
				dGeomCapsuleGetParams(g, &r, &l);
				dsDrawCapsule(pos, rot, static_cast<float>(l), static_cast<float>(r));
				break;
			case dCylinderClass:
				dGeomCylinderGetParams(g, &r, &l);
				dsDrawCylinder(pos, rot, static_cast<float>(l), static_cast<float>(r));
				break;
			case dTriMeshClass:
				dsDrawTriMesh1(pos, rot, mesh_index + meshCnt);
				meshCnt++;
				break;
			}
			g = dGeomGetBodyNext(g);
		}
	}
	if (HINGEAXIS)
	{
		for (int k = 0; k < 12; k++)
			rot[k] = 0.0f;
		rot[0] = 1.0f;
		rot[5] = 1.0f;
		rot[10] = 1.0f;
		for (int i = 0; i < joint_num; i++)
		{
			type = dJointGetType(joint_list[i]);
			switch (type)
			{
			case dJointTypeBall:
				dJointGetBallAnchor(joint_list[i], result1);
				dJointGetBallAnchor2(joint_list[i], result2);
				// draw anchor 1 in red
				pos[0] = static_cast<float>(result1[0]);
				pos[1] = static_cast<float>(result1[1]);
				pos[2] = static_cast<float>(result1[2]);
				dsSetColor(1.0f, 0.0f, 0.0f);
				dsDrawSphere(pos, rot, JointRadius);
				// draw anchor 2 in blue
				pos[0] = static_cast<float>(result2[0]);
				pos[1] = static_cast<float>(result2[1]);
				pos[2] = static_cast<float>(result2[2]);
				dsSetColor(0.0f, 0.0f, 1.0f);
				dsDrawSphere(pos, rot, JointRadius);
				break;
			case dJointTypeHinge:
				dJointGetHingeAnchor(joint_list[i], result1);
				dJointGetHingeAnchor2(joint_list[i], result2);
				pos[0] = static_cast<float>(result1[0]);
				pos[1] = static_cast<float>(result1[1]);
				pos[2] = static_cast<float>(result1[2]);
				// draw anchor 1 in red
				pos[0] = static_cast<float>(result1[0]);
				pos[1] = static_cast<float>(result1[1]);
				pos[2] = static_cast<float>(result1[2]);
				dsSetColor(1.0f, 0.0f, 0.0f);
				dsDrawSphere(pos, rot, JointRadius);
				// draw anchor 2 in blue
				pos[0] = static_cast<float>(result2[0]);
				pos[1] = static_cast<float>(result2[1]);
				pos[2] = static_cast<float>(result2[2]);
				dsSetColor(0.0f, 0.0f, 1.0f);
				dsDrawSphere(pos, rot, JointRadius);
				dJointGetHingeAxis1(joint_list[i], haxis_result);
				axis_norm = static_cast<float>(AxisLength / sqrt(haxis_result[0] * haxis_result[0] + haxis_result[1] * haxis_result[1] + haxis_result[2] * haxis_result[2]));
				pos1[0] = static_cast<float>(result1[0]);
				pos2[0] = static_cast<float>(pos1[0] + axis_norm * haxis_result[0]);
				pos1[1] = static_cast<float>(result1[1]);
				pos2[1] = static_cast<float>(pos1[1] + axis_norm * haxis_result[1]);
				pos1[2] = static_cast<float>(result1[2]);
				pos2[2] = static_cast<float>(pos1[2] + axis_norm * haxis_result[2]);
				dsSetColor(1.0f, 0.0f, 0.0f);
				dsDrawLine(pos1, pos2);
				dJointGetHingeAxis2(joint_list[i], haxis_result);
				axis_norm = static_cast<float>(AxisLength / sqrt(haxis_result[0] * haxis_result[0] + haxis_result[1] * haxis_result[1] + haxis_result[2] * haxis_result[2]));
				pos1[0] = static_cast<float>(result2[0]);
				pos2[0] = static_cast<float>(pos1[0] + axis_norm * haxis_result[0]);
				pos1[1] = static_cast<float>(result2[1]);
				pos2[1] = static_cast<float>(pos1[1] + axis_norm * haxis_result[1]);
				pos1[2] = static_cast<float>(result2[2]);
				pos2[2] = static_cast<float>(pos1[2] + axis_norm * haxis_result[2]);
				dsSetColor(0.0f, 0.0f, 1.0f);
				dsDrawLine(pos1, pos2);
				break;
			}
		}
	}

	if (LOCALAXIS)
	{
		for (int i = 0; i < draw_axis_num; i++)
		{
			dGeomCopyPosRot4Render(draw_axis_list[i], posr);
			for (int k = 0; k < 3; k++)
				pos[k] = static_cast<float>(posr->pos[k]);
			for (int k = 0; k < 12; k++)
				rot[k] = static_cast<float>(posr->R[k]);

			dsDrawLocalAxis(pos, rot, AxisLength);
		}
	}

	if (video_recorder.is_enable())
	{
		video_recorder.appendFrame();
	}
}

void dsDrawWorld()
{
	dsFunctions fn;
	fn.version = DS_VERSION;
	fn.start = &drawStart;
	fn.step = &dsDrawStep;
	fn.command = 0;
	fn.stop = 0;
	fn.path_to_textures = 0; // uses default

	// run simulation
	dsSimulationLoop(0, NULL, ds_window_width, ds_window_height, &fn);
}


void dsDrawWorldYesRender()
{
	dsFunctions fn;
	fn.version = DS_VERSION;
	fn.start = &drawStartYesRender; //
	fn.step = &dsDrawStepYesRender; //
	fn.command = 0;
	fn.stop = 0;
	fn.path_to_textures = 0; // uses default

	// run simulation
	dsSimulationLoop(0, NULL, ds_window_width, ds_window_height, &fn);
}

void dsDrawWorldWithMuscleWithRef()
{
	dsFunctions fn;
	fn.version = DS_VERSION;
	// fn.start = &drawStartYesRender; //
	// fn.step = &dsDrawStepYesRender; //
	fn.start = &drawStart;
	fn.step = &dsDrawStepWithMuscleWithRef;
	fn.command = 0;
	fn.stop = 0;
	fn.path_to_textures = 0; // uses default

	// run simulation
	dsSimulationLoop(0, NULL, ds_window_width, ds_window_height, &fn);
}


void dsDrawWorldOnlyMuscleWithRef()
{
	dsFunctions fn;
	fn.version = DS_VERSION;
	// fn.start = &drawStartYesRender; //
	// fn.step = &dsDrawStepYesRender; //
	fn.start = &drawStart;
	fn.step = &dsDrawStepOnlyMuscleWithRef;
	fn.command = 0;
	fn.stop = 0;
	fn.path_to_textures = 0; // uses default

	// run simulation
	dsSimulationLoop(0, NULL, ds_window_width, ds_window_height, &fn);
}


static std::unique_ptr<std::thread> thread_ptr;
void dsDrawWorldinThread()
{
	thread_ptr = std::unique_ptr<std::thread>(new std::thread(dsDrawWorld));
}


void dsDrawWorldinThreadYesRender()
{
	thread_ptr = std::unique_ptr<std::thread>(new std::thread(dsDrawWorldYesRender));
}


void dsDrawWorldinThreadWithMuscleWithRef()
{
	thread_ptr = std::unique_ptr<std::thread>(new std::thread(dsDrawWorldWithMuscleWithRef));
}


void dsDrawWorldinThreadOnlyMuscleWithRef()
{
	thread_ptr = std::unique_ptr<std::thread>(new std::thread(dsDrawWorldOnlyMuscleWithRef));
}


void dsTrackBodyWrapper(dBodyID target, int track_character, int sync_y)
{
	dsTrackBody(target, track_character, sync_y);
}

void dsCameraLookAtWrapper(float pos_x, float pos_y, float pos_z, float target_x, float target_y, float target_z, float up_x, float up_y, float up_z)
{
	dsCameraLookAt(pos_x, pos_y, pos_z, target_x, target_y, target_z, up_x, up_y, up_z);
}

void dsKillThread()
{
	dsKill();
	dsCallWindow();
	// join to quit render thread
	thread_ptr->join();
}

void dsSlowforRender()
{
	std::this_thread::sleep_for(std::chrono::milliseconds(pauseTime));
}

// Add by Zhenhua Song
void dsSetWindowWidth(int value)
{
	ds_window_width = value;
}

// Add by Zhenhua Song
void dsSetWindowHeight(int value)
{
	ds_window_height = value;
}

// Add by Zhenhua Song
int dsGetWindowWidth()
{
	return ds_window_width;
}

// Add by Zhenhua Song
int dsGetWindowHeight()
{
	return ds_window_height;
}

// Add by Zhenhua Song
void dsGetScreenBuffer(unsigned char *data)
{
	memcpy(data, ds_screen_buffer, sizeof(unsigned char) * 3 * ds_window_width * ds_window_height);
}

void dsStartRecordVideo()
{
	video_recorder.set_enable();
}

void dsPauseRecordVideo()
{
	video_recorder.set_disable();
}

size_t dsGetVideoFrame()
{
	return video_recorder.total_frame();
}

void dsEndRecordVideo(unsigned char *data, size_t num_frame)
{
	video_recorder.set_disable();
	if (data != nullptr)
	{
		size_t offset = sizeof(unsigned char) * 3 * ds_window_width * ds_window_height;
		for (size_t i = 0; i < num_frame; i++)
		{
			memcpy(data + i * offset, video_recorder.buffer[i], offset);
		}
	}
	video_recorder.clear();
}

#endif

#if !defined(WIN32)
void dsWorldSetter(dWorldID value)
{
}
dWorldID dsWorldGetter() { return NULL; }
void dsTrackBodyWrapper(dBodyID target, int track_character, int sync_y) {}
void dsAssignJointRadius(float x) {}
void dsAssignAxisLength(float x) {}
void dsCameraLookAtWrapper(float pos_x, float pos_y, float pos_z, float target_x, float target_y, float target_z, float up_x, float up_y, float up_z) {}
void dsAssignColor(float red, float green, float blue) {}
void dsAssignPauseTime(int value) {}
void dsAssignBackground(int x) {}
void dsWhetherHingeAxis(int x) {}
void dsWhetherLocalAxis(int x) {}
void dsDrawWorldinThread() {}

void dsDrawWorldinThreadYesRender() {}

void dsDrawWorldinThreadWithMuscleWithRef() {}

void dsDrawWorldinThreadOnlyMuscleWithRef() {}

void dsAssignMuscleHoldAnchorNum(int, int, int *) {}

void dsAssignMusclePartAndColor(int, int *, float *, double *) {}

void dsAssignMusclePartAndColorAll(int, int *, float *, double *, double *, double *, double *) {}

void dsAssignMuscleProperty(float *, double *) {}

void dsAssignMusclePropertyWithRef(float *, double *, double *) {}

void dsKillThread() {}
void dsSlowforRender() {}

void dsSetWindowWidth(int value) {}
void dsSetWindowHeight(int value) {}
int dsGetWindowWidth() {}
int dsGetWindowHeight() {}
void dsGetScreenBuffer(unsigned char *data) {}
void dsStartRecordVideo() {}
void dsPauseRecordVideo() {}
size_t dsGetVideoFrame() {}
void dsEndRecordVideo(unsigned char *output_data, size_t num_frame) {}
#endif
