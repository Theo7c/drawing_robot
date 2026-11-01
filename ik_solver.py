import numpy as np
L1 = 180.0
L2 = 180.0

paper_width = 297.0   
paper_height = 210.0  
border = 10.0         
space = 80.0          
"""
np.random.seed(42)
num_points = 100
raw_x = np.random.uniform(50, 1500, num_points)
raw_y = np.random.uniform(100, 900, num_points)
raw_coords = np.column_stack((raw_x, raw_y))
"""
raw_coords = np.load("frieren_path.npy")
num_points = len(raw_coords)
#offsettting so positive positive
min_x = raw_coords[:, 0].min()
min_y = raw_coords[:, 1].min()

norm_x = raw_coords[:, 0] - min_x
norm_y = raw_coords[:, 1] - min_y

span_x = norm_x.max()
span_y = norm_y.max()

#scale to paper dimenstions
printable_w = paper_width - 2 * border   
printable_h = paper_height - 2 * border  

scale = min(printable_w / span_x, printable_h / span_y)

scaled_x = norm_x * scale
scaled_y = norm_y * scale

#offset again so arm is at origin
max_scaled_x = scaled_x.max()

mapped_x = scaled_x - (max_scaled_x / 2.0)
mapped_y = scaled_y + border + space

coords = np.column_stack((mapped_x, mapped_y))

#translating x,y to joint angles
r_sq = mapped_x**2 + mapped_y**2
r = np.sqrt(r_sq)

#elbow angle using cos law
cos_theta2 = (r_sq - L1**2 - L2**2) / (2.0 * L1 * L2)
cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
theta2 = np.arccos(cos_theta2)

#shoulder base angle using sin law
sin_phi = (L2 * np.sin(theta2)) / r
sin_phi = np.clip(sin_phi, -1.0, 1.0)
phi = np.arcsin(sin_phi)

theta1 = np.arctan2(mapped_y, mapped_x) - phi

#reverses to find max error
x_fk = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
y_fk = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)

fk_error = np.max(np.hypot(mapped_x - x_fk, mapped_y - y_fk))

#print results
print(f"Total points generated: {num_points}")
print(f"Workspace Extents:")
print(f"  X span: [{mapped_x.min():.1f}, {mapped_x.max():.1f}] mm (width = {(mapped_x.max() - mapped_x.min()):.1f} mm)")
print(f"  Y span: [{mapped_y.min():.1f}, {mapped_y.max():.1f}] mm (height = {(mapped_y.max() - mapped_y.min()):.1f} mm)")
print(f"\nJoint Angle Ranges:")
print(f"  Shoulder (theta1): [{np.degrees(theta1.min()):.1f}°, {np.degrees(theta1.max()):.1f}°]")
print(f"  Elbow    (theta2): [{np.degrees(theta2.min()):.1f}°, {np.degrees(theta2.max()):.1f}°]")
print(f"\nFK Verification Max Error: {fk_error:.2e} mm")

#saves angles for two mirrored arms
joint_trajectory = np.column_stack((theta1, theta2,theta1+theta2,-theta2))

np.save("trajectory_angles.npy", joint_trajectory)

print("Saved joint trajectory to trajectory_angles.npy")
