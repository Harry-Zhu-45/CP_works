import vpython as vp


# Create a scene with a title and a size
scene = vp.canvas(title="Elements 3D", width=500, height=500, range=10)

# Create a 3D scene with a title and a size
vp.sphere(pos=vp.vector(0, 0, 0), radius=1, color=vp.color.green)
vp.sphere(pos=vp.vector(0, 1, -3), radius=1, color=vp.color.red)
vp.arrow(pos=vp.vector(3, 2, 2), axis=vp.vector(3, 1, 1), color=vp.color.cyan)
vp.cylinder(pos=vp.vector(-3, -2, 3), axis=vp.vector(6, -1, 5), color=vp.color.yellow)
vp.cone(pos=vp.vector(-6, -6, 0), axis=vp.vector(-2, 1, -0.5), radius=2, color=vp.color.magenta)
vp.helix(pos=vp.vector(-5, 5, -2), axis=vp.vector(5, 0, 0), radius=2, thickness=0.4, color=vp.color.orange)
vp.ring(pos=vp.vector(-6, 1, 0), axis=vp.vector(1, 1, 1), radius=2, thickness=0.3, color=vp.vector(0.3, 0.4, 0.6))
vp.box(pos=vp.vector(5, -2, 2), length=5, width=5, height=0.4, color=vp.vector(0.4, 0.8, 0.2))
vp.pyramid(pos=vp.vector(2, 5, 2), size=vp.vector(4, 3, 2), color=vp.vector(0.7, 0.7, 0.2))
vp.ellipsoid(pos=vp.vector(-1, -7, 1), axis=vp.vector(2, 1, 3), length=4, height=2, width=5, color=vp.vector(0.1, 0.9, 0.8))
