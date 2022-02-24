# -*- coding: utf-8 -*-
import hydraulic

print(round(3000e-6 * hydraulic.WaterPressure(.2), 4), "N")
print(round(hydraulic.Buoyancy(5.2e-5), 4), "N")
print(round(hydraulic.Buoyancy(1.5e-5), 4), "N")
print(round(hydraulic.ReynoldsNumberRoundPipe(0.02, 0.12, 0.013e-4), 4), "")
print(round(hydraulic.FrictionalDragCoefficientRoundPipe(1840), 4), "")
print(round(hydraulic.HeadLossRoundPipe(0.0348,20,0.12,0.02), 4), "m")
