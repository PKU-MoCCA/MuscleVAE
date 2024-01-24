/*************************************************************************
 *                                                                       *
 * Open Dynamics Engine, Copyright (C) 2001,2002 Russell L. Smith.       *
 * All rights reserved.  Email: russ@q12.org   Web: www.q12.org          *
 *                                                                       *
 * This library is free software; you can redistribute it and/or         *
 * modify it under the terms of EITHER:                                  *
 *   (1) The GNU Lesser General Public License as published by the Free  *
 *       Software Foundation; either version 2.1 of the License, or (at  *
 *       your option) any later version. The text of the GNU Lesser      *
 *       General Public License is included with this library in the     *
 *       file LICENSE.TXT.                                               *
 *   (2) The BSD-style license that is included with this library in     *
 *       the file LICENSE-BSD.TXT.                                       *
 *                                                                       *
 * This library is distributed in the hope that it will be useful,       *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the files    *
 * LICENSE.TXT and LICENSE-BSD.TXT for more details.                     *
 *                                                                       *
 *************************************************************************/

/*

quaternions have the format: (s,vx,vy,vz) where (vx,vy,vz) is the
"rotation axis" and s is the "rotation angle".

*/

#include <ode/rotation.h>
#include <ode/odemath.h>
#include "config.h"


#define _R(i,j) R[(i)*4+(j)]

#define SET_3x3_IDENTITY \
  _R(0,0) = REAL(1.0); \
  _R(0,1) = REAL(0.0); \
  _R(0,2) = REAL(0.0); \
  _R(0,3) = REAL(0.0); \
  _R(1,0) = REAL(0.0); \
  _R(1,1) = REAL(1.0); \
  _R(1,2) = REAL(0.0); \
  _R(1,3) = REAL(0.0); \
  _R(2,0) = REAL(0.0); \
  _R(2,1) = REAL(0.0); \
  _R(2,2) = REAL(1.0); \
  _R(2,3) = REAL(0.0);


void dRSetIdentity (dMatrix3 R)
{
  dAASSERT (R);
  SET_3x3_IDENTITY;
}


void dRFromAxisAndAngle (dMatrix3 R, dReal ax, dReal ay, dReal az,
			 dReal angle)
{
  dAASSERT (R);
  dQuaternion q;
  dQFromAxisAndAngle (q,ax,ay,az,angle);
  dQtoR (q,R);
}


void dRFromEulerAngles (dMatrix3 R, dReal phi, dReal theta, dReal psi)
{
  dReal sphi,cphi,stheta,ctheta,spsi,cpsi;
  dAASSERT (R);
  sphi = dSin(phi);
  cphi = dCos(phi);
  stheta = dSin(theta);
  ctheta = dCos(theta);
  spsi = dSin(psi);
  cpsi = dCos(psi);
  _R(0,0) = cpsi*ctheta;
  _R(0,1) = spsi*ctheta;
  _R(0,2) =-stheta;
  _R(0,3) = REAL(0.0);
  _R(1,0) = cpsi*stheta*sphi - spsi*cphi;
  _R(1,1) = spsi*stheta*sphi + cpsi*cphi;
  _R(1,2) = ctheta*sphi;
  _R(1,3) = REAL(0.0);
  _R(2,0) = cpsi*stheta*cphi + spsi*sphi;
  _R(2,1) = spsi*stheta*cphi - cpsi*sphi;
  _R(2,2) = ctheta*cphi;
  _R(2,3) = REAL(0.0);
}


void dRFrom2Axes (dMatrix3 R, dReal ax, dReal ay, dReal az, dReal bx, dReal by, dReal bz)
{
  dReal l,k;
  dAASSERT (R);
  l = dSqrt (ax*ax + ay*ay + az*az);
  if (l <= REAL(0.0)) {
    dDEBUGMSG ("zero length vector");
    return;
  }
  l = dRecip(l);
  ax *= l;
  ay *= l;
  az *= l;
  k = ax*bx + ay*by + az*bz;
  bx -= k*ax;
  by -= k*ay;
  bz -= k*az;
  l = dSqrt (bx*bx + by*by + bz*bz);
  if (l <= REAL(0.0)) {
    dDEBUGMSG ("zero length vector");
    return;
  }
  l = dRecip(l);
  bx *= l;
  by *= l;
  bz *= l;
  _R(0,0) = ax;
  _R(1,0) = ay;
  _R(2,0) = az;
  _R(0,1) = bx;
  _R(1,1) = by;
  _R(2,1) = bz;
  _R(0,2) = - by*az + ay*bz;
  _R(1,2) = - bz*ax + az*bx;
  _R(2,2) = - bx*ay + ax*by;
  _R(0,3) = REAL(0.0);
  _R(1,3) = REAL(0.0);
  _R(2,3) = REAL(0.0);
}


void dRFromZAxis (dMatrix3 R, dReal ax, dReal ay, dReal az)
{
  dVector3 n,p,q;
  n[0] = ax;
  n[1] = ay;
  n[2] = az;
  dNormalize3 (n);
  dPlaneSpace (n,p,q);
  _R(0,0) = p[0];
  _R(1,0) = p[1];
  _R(2,0) = p[2];
  _R(0,1) = q[0];
  _R(1,1) = q[1];
  _R(2,1) = q[2];
  _R(0,2) = n[0];
  _R(1,2) = n[1];
  _R(2,2) = n[2];
  _R(0,3) = REAL(0.0);
  _R(1,3) = REAL(0.0);
  _R(2,3) = REAL(0.0);
}


void dQSetIdentity (dQuaternion q)
{
  dAASSERT (q);
  q[0] = 1;
  q[1] = 0;
  q[2] = 0;
  q[3] = 0;
}


void dQFromAxisAndAngle (dQuaternion q, dReal ax, dReal ay, dReal az, dReal angle)
{
  dAASSERT (q);
  dReal l = ax*ax + ay*ay + az*az;
  if (l > REAL(0.0)) {
    angle *= REAL(0.5);
    q[0] = dCos (angle);
    l = dSin(angle) * dRecipSqrt(l);
    q[1] = ax*l;
    q[2] = ay*l;
    q[3] = az*l;
  }
  else {
    q[0] = 1;
    q[1] = 0;
    q[2] = 0;
    q[3] = 0;
  }
}


void dQMultiply0 (dQuaternion qa, const dQuaternion qb, const dQuaternion qc)
{
  dAASSERT (qa && qb && qc);
  qa[0] = qb[0]*qc[0] - qb[1]*qc[1] - qb[2]*qc[2] - qb[3]*qc[3];
  qa[1] = qb[0]*qc[1] + qb[1]*qc[0] + qb[2]*qc[3] - qb[3]*qc[2];
  qa[2] = qb[0]*qc[2] + qb[2]*qc[0] + qb[3]*qc[1] - qb[1]*qc[3];
  qa[3] = qb[0]*qc[3] + qb[3]*qc[0] + qb[1]*qc[2] - qb[2]*qc[1];
}

// inv(qb) * qc
void dQMultiply1 (dQuaternion qa, const dQuaternion qb, const dQuaternion qc)
{
  dAASSERT (qa && qb && qc);
  qa[0] = qb[0]*qc[0] + qb[1]*qc[1] + qb[2]*qc[2] + qb[3]*qc[3];
  qa[1] = qb[0]*qc[1] - qb[1]*qc[0] - qb[2]*qc[3] + qb[3]*qc[2];
  qa[2] = qb[0]*qc[2] - qb[2]*qc[0] - qb[3]*qc[1] + qb[1]*qc[3];
  qa[3] = qb[0]*qc[3] - qb[3]*qc[0] - qb[1]*qc[2] + qb[2]*qc[1];
}

// qb * inv(qc)
void dQMultiply2 (dQuaternion qa, const dQuaternion qb, const dQuaternion qc)
{
  dAASSERT (qa && qb && qc);
  qa[0] =  qb[0]*qc[0] + qb[1]*qc[1] + qb[2]*qc[2] + qb[3]*qc[3];
  qa[1] = -qb[0]*qc[1] + qb[1]*qc[0] - qb[2]*qc[3] + qb[3]*qc[2];
  qa[2] = -qb[0]*qc[2] + qb[2]*qc[0] - qb[3]*qc[1] + qb[1]*qc[3];
  qa[3] = -qb[0]*qc[3] + qb[3]*qc[0] - qb[1]*qc[2] + qb[2]*qc[1];
}

// inv(qb) * inv(qc)
void dQMultiply3 (dQuaternion qa, const dQuaternion qb, const dQuaternion qc)
{
  dAASSERT (qa && qb && qc);
  qa[0] =  qb[0]*qc[0] - qb[1]*qc[1] - qb[2]*qc[2] - qb[3]*qc[3];
  qa[1] = -qb[0]*qc[1] - qb[1]*qc[0] + qb[2]*qc[3] - qb[3]*qc[2];
  qa[2] = -qb[0]*qc[2] - qb[2]*qc[0] + qb[3]*qc[1] - qb[1]*qc[3];
  qa[3] = -qb[0]*qc[3] - qb[3]*qc[0] + qb[1]*qc[2] - qb[2]*qc[1];
}


// dRfromQ(), dQfromR() and dDQfromW() are derived from equations in "An Introduction
// to Physically Based Modeling: Rigid Body Simulation - 1: Unconstrained
// Rigid Body Dynamics" by David Baraff, Robotics Institute, Carnegie Mellon
// University, 1997.

void dRfromQ (dMatrix3 R, const dQuaternion q)
{
  dAASSERT (q && R);
  // q = (s,vx,vy,vz)
  dReal qq1 = 2*q[1]*q[1];
  dReal qq2 = 2*q[2]*q[2];
  dReal qq3 = 2*q[3]*q[3];
  _R(0,0) = 1 - qq2 - qq3;
  _R(0,1) = 2*(q[1]*q[2] - q[0]*q[3]);
  _R(0,2) = 2*(q[1]*q[3] + q[0]*q[2]);
  _R(0,3) = REAL(0.0);
  _R(1,0) = 2*(q[1]*q[2] + q[0]*q[3]);
  _R(1,1) = 1 - qq1 - qq3;
  _R(1,2) = 2*(q[2]*q[3] - q[0]*q[1]);
  _R(1,3) = REAL(0.0);
  _R(2,0) = 2*(q[1]*q[3] - q[0]*q[2]);
  _R(2,1) = 2*(q[2]*q[3] + q[0]*q[1]);
  _R(2,2) = 1 - qq1 - qq2;
  _R(2,3) = REAL(0.0);
}


void dQfromR (dQuaternion q, const dMatrix3 R)
{
  dAASSERT (q && R);
  dReal tr,s;
  tr = _R(0,0) + _R(1,1) + _R(2,2);
  if (tr >= 0) {
    s = dSqrt (tr + 1);
    q[0] = REAL(0.5) * s;
    s = REAL(0.5) * dRecip(s);
    q[1] = (_R(2,1) - _R(1,2)) * s;
    q[2] = (_R(0,2) - _R(2,0)) * s;
    q[3] = (_R(1,0) - _R(0,1)) * s;
  }
  else {
    // find the largest diagonal element and jump to the appropriate case
    if (_R(1,1) > _R(0,0)) {
      if (_R(2,2) > _R(1,1)) goto case_2;
      goto case_1;
    }
    if (_R(2,2) > _R(0,0)) goto case_2;
    goto case_0;

    case_0:
    s = dSqrt((_R(0,0) - (_R(1,1) + _R(2,2))) + 1);
    q[1] = REAL(0.5) * s;
    s = REAL(0.5) * dRecip(s);
    q[2] = (_R(0,1) + _R(1,0)) * s;
    q[3] = (_R(2,0) + _R(0,2)) * s;
    q[0] = (_R(2,1) - _R(1,2)) * s;
    return;

    case_1:
    s = dSqrt((_R(1,1) - (_R(2,2) + _R(0,0))) + 1);
    q[2] = REAL(0.5) * s;
    s = REAL(0.5) * dRecip(s);
    q[3] = (_R(1,2) + _R(2,1)) * s;
    q[1] = (_R(0,1) + _R(1,0)) * s;
    q[0] = (_R(0,2) - _R(2,0)) * s;
    return;

    case_2:
    s = dSqrt((_R(2,2) - (_R(0,0) + _R(1,1))) + 1);
    q[3] = REAL(0.5) * s;
    s = REAL(0.5) * dRecip(s);
    q[1] = (_R(2,0) + _R(0,2)) * s;
    q[2] = (_R(1,2) + _R(2,1)) * s;
    q[0] = (_R(1,0) - _R(0,1)) * s;
    return;
  }
}


void dDQfromW (dReal dq[4], const dVector3 w, const dQuaternion q)
{
  dAASSERT (w && q && dq);
  dq[0] = REAL(0.5)*(- w[0]*q[1] - w[1]*q[2] - w[2]*q[3]);
  dq[1] = REAL(0.5)*(  w[0]*q[0] + w[1]*q[3] - w[2]*q[2]);
  dq[2] = REAL(0.5)*(- w[0]*q[3] + w[1]*q[0] + w[2]*q[1]);
  dq[3] = REAL(0.5)*(  w[0]*q[2] - w[1]*q[1] + w[2]*q[0]);
}

// Add by Zhenhua Song
void dQBetweenVec(const dVector3 a, const dVector3 b, dQuaternion result)
{
    dReal a_dot_b = dCalcVectorDot3(a, b);
    dReal a_sqr = dCalcVectorLengthSquare3(a);
    dReal b_sqr = dCalcVectorLengthSquare3(b);
    dReal w = dSqrt(a_sqr * b_sqr) + a_dot_b; // w componet of quaternion.

    dVector3 a_cross_b;
    dCalcVectorCross3(a_cross_b, a, b);  // axis of quaternion

    result[0] = w;
    result[1] = a_cross_b[0];
    result[2] = a_cross_b[1];
    result[3] = a_cross_b[2];

    dNormalize4(result);
}


// Using XYZ order
void dRxRyRzfromR_XYZ(dMatrix3 Rx, dMatrix3 Ry, dMatrix3 Rz, const dMatrix3 R)
{
  // +-           -+   +-                                        -+
  // | r00 r01 r02 |   |  cy*cz           -cy*sz            sy    |
  // | r10 r11 r12 | = |  cz*sx*sy+cx*sz   cx*cz-sx*sy*sz  -cy*sx |
  // | r20 r21 r22 |   | -cx*cz*sy+sx*sz   cz*sx+cx*sy*sz   cx*cy |
  // +-           -+   +-                                        -+
  dAASSERT(Rx && Ry && Rz && R);
  dReal cy = sqrt(_R(0, 0) * _R(0, 0) + _R(0, 1) * _R(0, 1));
  bool singular = cy < 1e-6;
  dReal x, y, z;
  if (_R(0, 2) < 1.0)
  {
    if (_R(0, 2) > -1.0)
    {
      // x_angle = atan2(-r12,r22)
      // y_angle = asin(r02)
      // z_angle = atan2(-r01,r00)
      x = atan2(-_R(1, 2), _R(2, 2));
      y = asin(_R(0, 2));
      z = atan2(-_R(0, 1), _R(0, 0));
    }
    else
    {
      // z_angle - x_angle = atan2(r10,r11)
      // y_angle = -pi/2
      // WARNING.  The solution is not unique.  Choosing z_angle = 0.
      x = -atan2(_R(1, 0), _R(1, 1));
      y = -M_PI_2;
      z = 0.0;
    }
  }
  else
  {
    // z_angle + x_angle = atan2(r10,r11)
    // y_angle = +pi/2
    // WARNING.  The solutions is not unique.  Choosing z_angle = 0.
    x = -atan2(_R(1, 0), _R(1, 1));
    y = M_PI_2;
    z = 0.0;
  }
  Rx[0] = 1.0;
  Rx[1] = 0.0;
  Rx[2] = 0.0;
  Rx[3] = 0.0;
  Rx[4] = 0.0;
  Rx[5] = cos(x);
  Rx[6] = -sin(x);
  Rx[7] = 0.0;
  Rx[8] = 0.0;
  Rx[9] = sin(x);
  Rx[10] = cos(x);
  Rx[11] = 0.0;

  Ry[0] = cos(y);
  Ry[1] = 0.0;
  Ry[2] = sin(y);
  Ry[3] = 0.0;
  Ry[4] = 0.0;
  Ry[5] = 1.0;
  Ry[6] = 0.0;
  Ry[7] = 0.0;
  Ry[8] = -sin(y);
  Ry[9] = 0.0;
  Ry[10] = cos(y);
  Ry[11] = 0.0;

  Rz[0] = cos(z);
  Rz[1] = -sin(z);
  Rz[2] = 0.0;
  Rz[3] = 0.0;
  Rz[4] = sin(z);
  Rz[5] = cos(z);
  Rz[6] = 0.0;
  Rz[7] = 0.0;
  Rz[8] = 0.0;
  Rz[9] = 0.0;
  Rz[10] = 1.0;
  Rz[11] = 0.0;
  // std::cout << x << " " << y << " " << z << std::endl;
}


void dAxAyAzfromRRp_format_dMatrix3(dMatrix3 AxAyAz, const dMatrix3 R, const dMatrix3 Rp)
{
  // Maybe WRONG ORDER!
  // S = [      1,       0,          0
  //         Rp 0, Rp Rx 1, Rp Rx Ry 0
  //            0,       0,          1]
  dVector3 Ax0 = {1.0, 0.0, 0.0, 0.0};
  dVector3 Ay0 = {0.0, 1.0, 0.0, 0.0};
  dVector3 Az0 = {0.0, 0.0, 1.0, 0.0};
  dAASSERT(Rp && R && AxAyAz);
  dMatrix3 Rx, Ry, Rz, Ry_, Rz_;
  dVector3 Ax, Ay, Az;
  dRxRyRzfromR_XYZ(Rx, Ry, Rz, R);
  simpleMatMul(Ry_, Rp, Rx);
  simpleMatMul(Rz_, Ry_, Ry);
  simpleMatMulVec(Ax, Rp, Ax0);
  simpleMatMulVec(Ay, Ry_, Ay0);
  simpleMatMulVec(Az, Rz_, Az0);
  AxAyAz[0] = Ax[0];
  AxAyAz[1] = Ax[1];
  AxAyAz[2] = Ax[2];
  AxAyAz[3] = 0.0;
  AxAyAz[4] = Ay[0];
  AxAyAz[5] = Ay[1];
  AxAyAz[6] = Ay[2];
  AxAyAz[7] = 0.0;
  AxAyAz[8] = Az[0];
  AxAyAz[9] = Az[1];
  AxAyAz[10] = Az[2];
  AxAyAz[11] = 0.0;
}


void dAxAyAzfromRRp_format_dMatrix3_dart_revised(dMatrix3 AxAyAz, const dMatrix3 R, const dMatrix3 Rp)
{
  // DART ORDER!
  // S = [            1,       0,    0
  //        Rp Rz Ry  0, Rp Rz 1, Rp 0
  //                  0,       0,    1]
  dVector3 Ax0 = {1.0, 0.0, 0.0, 0.0};
  dVector3 Ay0 = {0.0, 1.0, 0.0, 0.0};
  dVector3 Az0 = {0.0, 0.0, 1.0, 0.0};
  dAASSERT(Rp && R && AxAyAz);
  dMatrix3 Rx, Ry, Rz, Ry_, Rx_;
  dVector3 Ax, Ay, Az;
  dRxRyRzfromR_XYZ(Rx, Ry, Rz, R);
  simpleMatMul(Ry_, Rp, Rz);
  simpleMatMul(Rx_, Ry_, Ry);
  simpleMatMulVec(Ax, Rx_, Ax0);
  simpleMatMulVec(Ay, Ry_, Ay0);
  simpleMatMulVec(Az, Rp, Az0);
  AxAyAz[0] = Ax[0];
  AxAyAz[1] = Ax[1];
  AxAyAz[2] = Ax[2];
  AxAyAz[3] = 0.0;
  AxAyAz[4] = Ay[0];
  AxAyAz[5] = Ay[1];
  AxAyAz[6] = Ay[2];
  AxAyAz[7] = 0.0;
  AxAyAz[8] = Az[0];
  AxAyAz[9] = Az[1];
  AxAyAz[10] = Az[2];
  AxAyAz[11] = 0.0;
}


void dAxAyAzfromRRp(dVector3 Ax, dVector3 Ay, dVector3 Az, const dMatrix3 R, const dMatrix3 Rp)
{
  // Maybe WRONG ORDER!
  // S = [      1,       0,          0
  //         Rp 0, Rp Rx 1, Rp Rx Ry 0
  //            0,       0,          1]
  dVector3 Ax0 = {1.0, 0.0, 0.0, 0.0};
  dVector3 Ay0 = {0.0, 1.0, 0.0, 0.0};
  dVector3 Az0 = {0.0, 0.0, 1.0, 0.0};
  dAASSERT(Rp && R && Ax && Ay && Az);
  dMatrix3 Rx, Ry, Rz, Ry_, Rz_;
  dRxRyRzfromR_XYZ(Rx, Ry, Rz, R);
  simpleMatMul(Ry_, Rp, Rx);
  simpleMatMul(Rz_, Ry_, Ry);
  simpleMatMulVec(Ax, Rp, Ax0);
  simpleMatMulVec(Ay, Ry_, Ay0);
  simpleMatMulVec(Az, Rz_, Az0);
}


void dAxAyAzfromRRp_dart_revised(dVector3 Ax, dVector3 Ay, dVector3 Az, const dMatrix3 R, const dMatrix3 Rp)
{
  // DART ORDER!
  // S = [            1,       0,    0
  //        Rp Rz Ry  0, Rp Rz 1, Rp 0
  //                  0,       0,    1]
  dVector3 Ax0 = {1.0, 0.0, 0.0, 0.0};
  dVector3 Ay0 = {0.0, 1.0, 0.0, 0.0};
  dVector3 Az0 = {0.0, 0.0, 1.0, 0.0};
  dAASSERT(Rp && R && Ax && Ay && Az);
  dMatrix3 Rx, Ry, Rz, Ry_, Rx_;
  dRxRyRzfromR_XYZ(Rx, Ry, Rz, R);
  simpleMatMul(Ry_, Rp, Rz);
  simpleMatMul(Rx_, Ry_, Ry);
  simpleMatMulVec(Ax, Rx_, Ax0);
  simpleMatMulVec(Ay, Ry_, Ay0);
  simpleMatMulVec(Az, Rp, Az0);
}


void dEulerAnglefromR_XYZ(dVector3 angle, const dMatrix3 R)
{
  // +-           -+   +-                                        -+
  // | r00 r01 r02 |   |  cy*cz           -cy*sz            sy    |
  // | r10 r11 r12 | = |  cz*sx*sy+cx*sz   cx*cz-sx*sy*sz  -cy*sx |
  // | r20 r21 r22 |   | -cx*cz*sy+sx*sz   cz*sx+cx*sy*sz   cx*cy |
  // +-           -+   +-                                        -+
  dAASSERT(angle && R);
  dReal x, y, z;
  if (_R(0, 2) < 1.0)
  {
    if (_R(0, 2) > -1.0)
    {
      // x_angle = atan2(-r12,r22)
      // y_angle = asin(r02)
      // z_angle = atan2(-r01,r00)
      x = atan2(-_R(1, 2), _R(2, 2));
      y = asin(_R(0, 2));
      z = atan2(-_R(0, 1), _R(0, 0));
    }
    else
    {
      // z_angle - x_angle = atan2(r10,r11)
      // y_angle = -pi/2
      // WARNING.  The solution is not unique.  Choosing z_angle = 0.
      x = -atan2(_R(1, 0), _R(1, 1));
      y = -M_PI_2;
      z = 0.0;
    }
  }
  else
  {
    // z_angle + x_angle = atan2(r10,r11)
    // y_angle = +pi/2
    // WARNING.  The solutions is not unique.  Choosing z_angle = 0.
    x = -atan2(_R(1, 0), _R(1, 1));
    y = M_PI_2;
    z = 0.0;
  }
  angle[0] = x;
  angle[1] = y;
  angle[2] = z;
  angle[3] = 0.0;
}


void simpleMatMul(dMatrix3 res, const dMatrix3 Ra, const dMatrix3 Rb)
{
  dAASSERT(Ra && Rb && res);
  res[0] = Ra[0] * Rb[0] + Ra[1] * Rb[4] + Ra[2] * Rb[8];
  res[1] = Ra[0] * Rb[1] + Ra[1] * Rb[5] + Ra[2] * Rb[9];
  res[2] = Ra[0] * Rb[2] + Ra[1] * Rb[6] + Ra[2] * Rb[10];
  res[3] = 0.0;
  res[4] = Ra[4] * Rb[0] + Ra[5] * Rb[4] + Ra[6] * Rb[8];
  res[5] = Ra[4] * Rb[1] + Ra[5] * Rb[5] + Ra[6] * Rb[9];
  res[6] = Ra[4] * Rb[2] + Ra[5] * Rb[6] + Ra[6] * Rb[10];
  res[7] = 0.0;
  res[8] = Ra[8] * Rb[0] + Ra[9] * Rb[4] + Ra[10] * Rb[8];
  res[9] = Ra[8] * Rb[1] + Ra[9] * Rb[5] + Ra[10] * Rb[9];
  res[10] = Ra[8] * Rb[2] + Ra[9] * Rb[6] + Ra[10] * Rb[10];
  res[11] = 0.0;
}


void simpleMatMulVec(dMatrix3 res, const dMatrix3 R, const dVector3 vec)
{
  dAASSERT(R && vec && res);
  res[0] = _R(0, 0) * vec[0] + _R(0, 1) * vec[1] + _R(0, 2) * vec[2];
  res[1] = _R(1, 0) * vec[0] + _R(1, 1) * vec[1] + _R(1, 2) * vec[2];
  res[2] = _R(2, 0) * vec[0] + _R(2, 1) * vec[1] + _R(2, 2) * vec[2];
  res[3] = 0.0;
}


void dRelAxAyAzfromR(dVector3 Ax, dVector3 Ay, dVector3 Az, const dMatrix3 R)
{
  // Maybe WRONG ORDER!
  // S = [    1,    0,       0
  //          0, Rx 1, Rx Ry 0
  //          0,    0,       1]
  dVector3 Ax0 = {1.0, 0.0, 0.0, 0.0};
  dVector3 Ay0 = {0.0, 1.0, 0.0, 0.0};
  dVector3 Az0 = {0.0, 0.0, 1.0, 0.0};
  dAASSERT(R && Ax && Ay && Az);
  dMatrix3 Rx, Ry, Rz, Rz_;
  dRxRyRzfromR_XYZ(Rx, Ry, Rz, R);
  simpleMatMul(Rz_, Rx, Ry);
  Ax[0] = Ax0[0];
  Ax[1] = Ax0[1];
  Ax[2] = Ax0[2];
  Ax[3] = Ax0[3];
  simpleMatMulVec(Ay, Rx, Ay0);
  simpleMatMulVec(Az, Rz_, Az0);
}


void dRelAxAyAzfromR_dart_revised(dVector3 Ax, dVector3 Ay, dVector3 Az, const dMatrix3 R)
{
  // DART ORDER!
  // S = [         1,    0, 0
  //        Rz Ry  0, Rz 1, 0
  //               0,    0, 1]
  // S = [    c1*c2, s2,  0
  //       -(c1*s2), c2,  0
  //             s1,  0,  1 ]
  dVector3 Ax0 = {1.0, 0.0, 0.0, 0.0};
  dVector3 Ay0 = {0.0, 1.0, 0.0, 0.0};
  dVector3 Az0 = {0.0, 0.0, 1.0, 0.0};
  dAASSERT(R && Ax && Ay && Az);
  dMatrix3 Rx, Ry, Rz, Rx_;
  dRxRyRzfromR_XYZ(Rx, Ry, Rz, R);
  simpleMatMul(Rx_, Rz, Ry);
  Az[0] = Az0[0];
  Az[1] = Az0[1];
  Az[2] = Az0[2];
  Az[3] = Az0[3];
  simpleMatMulVec(Ay, Rz, Ay0);
  simpleMatMulVec(Ax, Rx_, Ax0);
}


void dRelAxAyAzfromR_format_dMatrix3(dMatrix3 AxAyAz, const dMatrix3 R)
{
  // Maybe WRONG ORDER!
  // S = [    1,    0,       0
  //          0, Rx 1, Rx Ry 0
  //          0,    0,       1]
  dVector3 Ax0 = {1.0, 0.0, 0.0, 0.0};
  dVector3 Ay0 = {0.0, 1.0, 0.0, 0.0};
  dVector3 Az0 = {0.0, 0.0, 1.0, 0.0};
  dAASSERT(R && AxAyAz);
  dMatrix3 Rx, Ry, Rz, Rz_;
  dRxRyRzfromR_XYZ(Rx, Ry, Rz, R);
  simpleMatMul(Rz_, Rx, Ry);
  dVector3 Ay, Az;
  simpleMatMulVec(Ay, Rx, Ay0);
  simpleMatMulVec(Az, Rz_, Az0);
  AxAyAz[0] = Ax0[0];
  AxAyAz[1] = Ax0[1];
  AxAyAz[2] = Ax0[2];
  AxAyAz[3] = 0.0;
  AxAyAz[4] = Ay[0];
  AxAyAz[5] = Ay[1];
  AxAyAz[6] = Ay[2];
  AxAyAz[7] = 0.0;
  AxAyAz[8] = Az[0];
  AxAyAz[9] = Az[1];
  AxAyAz[10] = Az[2];
  AxAyAz[11] = 0.0;
}


void dRelAxAyAzfromR_format_dMatrix3_dart_revised(dMatrix3 AxAyAz, const dMatrix3 R)
{
  // DART ORDER!
  // S = [         1,    0, 0
  //        Rz Ry  0, Rz 1, 0
  //               0,    0, 1]
  // S = [    c1*c2, s2,  0
  //       -(c1*s2), c2,  0
  //             s1,  0,  1 ]
  dVector3 Ax0 = {1.0, 0.0, 0.0, 0.0};
  dVector3 Ay0 = {0.0, 1.0, 0.0, 0.0};
  dVector3 Az0 = {0.0, 0.0, 1.0, 0.0};
  dAASSERT(R && AxAyAz);
  dMatrix3 Rx, Ry, Rz, Rx_;
  dRxRyRzfromR_XYZ(Rx, Ry, Rz, R);
  simpleMatMul(Rx_, Rz, Ry);
  dVector3 Ay, Ax;
  simpleMatMulVec(Ay, Rz, Ay0);
  simpleMatMulVec(Ax, Rx_, Ax0);
  AxAyAz[0] = Ax[0];
  AxAyAz[1] = Ax[1];
  AxAyAz[2] = Ax[2];
  AxAyAz[3] = 0.0;
  AxAyAz[4] = Ay[0];
  AxAyAz[5] = Ay[1];
  AxAyAz[6] = Ay[2];
  AxAyAz[7] = 0.0;
  AxAyAz[8] = Az0[0];
  AxAyAz[9] = Az0[1];
  AxAyAz[10] = Az0[2];
  AxAyAz[11] = 0.0;
}
