# -*- coding: utf-8 -*-
standardAtmosphericPressure = 1.01325e5  # 1标准大气压 1.01325×10^5Pa
waterDensity = 0.998232e3  # 水的密度ρ为0.998232×10^3kg/m^3  20度
volumetricWeight = 9790  # 水的容重γ=ρg（读音gamma） 9790N/m^3
gravity = 9.807  # 重力加速度取9.807 N/kg

# μ 粘滞系数 coefficient of viscosity Pas（帕秒）
# σ 表面张力系数 Surface tension coefficient  N/m
"""
水压：p=ρgh（p是压强，ρ是液体密度，h是取压点到液面高度）
height 单位 m
返回单位 N/m^2
"""


def WaterPressure(height):
    return waterDensity * gravity * height


"""
浮力:F浮=G排（F=ρgV）
volume 单位 m^3
返回单位 N
"""


def Buoyancy(volume):
    return waterDensity * gravity * volume


"""
流量:Q=AV
area, velocity 单位m^2，m/s，
返回单位 m^3/s
"""


def FlowRate(area, velocity):
    return area*velocity


"""
伯努利原理 Bernoulli's principle
p+1/2ρv^2+ρgh=C
p/γ+Z+v^2/2g=C

p 静压， 1/2ρv^2 动压 , C  总压
Z 位置水头，单位位能
p/γ 压强水头，单位压强
v^2/2g 流速水头，单位动能
C 总水头，单位总能量
"""

"""
雷诺系数 Reynolds number
一般管道雷诺数Re<2000为层流状态，Re>4000为紊流状态，Re=2000～4000为过渡状态

圆管公式 Re=vd/γ
d 直径 m
v 流速 m/s
γ 运动黏滞系数 m^2/s
"""


def ReynoldsNumberRoundPipe(diameter, velocity, KinematicViscosityCoefficient):
    return diameter*velocity/KinematicViscosityCoefficient


"""
沿程阻力（Frictional Drag）：流体流经一定管径的直管时，由于流体内摩擦力而产生的阻力

圆管层流时：
λ=64/Re
hf=λ*l*v^2/2gd
λ 沿程阻力系数
hf 水头损失 m
l 直管长度 m
v 流速 m/s
g 重力加速度
d 直径 m
"""


def FrictionalDragCoefficientRoundPipe(ReynoldsNumber):
    return 64/ReynoldsNumber


def HeadLossRoundPipe(frictionalDrag, lenght, velocity, diameter):
    return frictionalDrag*lenght*velocity*velocity/2/gravity/diameter


"""
雷诺系数 Reynolds number
异型管公式 Re=4vA/γS
A 面积 m^2
S 周长 m
v 流速 m/s
γ 运动黏滞系数 m^2/s
"""


def ReynoldsNumber(area, perimeter, velocity, KinematicViscosityCoefficient):
    return 4*area*velocity/perimeter/KinematicViscosityCoefficient
