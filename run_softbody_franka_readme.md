# Readme
cd /home/goodmansun/newton
source .venv/bin/activate
python -m newton.examples softbody_franka  

example_softbody_franka.py(line: 357)
`Contacts`中有软体接触的几何信息：接触数量，接触shape，接触法线等（contacts.py(line 263)），但是没有直接提供求解后的力。

在这些example中有使用MuJoCo solver的地方。example_sensor_contact.py(line 125)。而`example_softbody_franka`使用的是`SolverVBD`。它没有提供Newton所需要的公共力更新接口

好消息是软体性质是可以自定义的。add_soft_mesh() 支持 TetMesh，也支持直接传 vertices + tetra indices；k_mu、k_lambda、k_damp 还能是每个四面体不同的数组：￼builder.py (line 8349)。规则上它需要的是四面体体网格，不是普通表面 mesh。普通 .obj/.stl 这类表面模型通常要先 tetrahedralize。

常改的参数：

```
scene.add_soft_mesh(
    pos=...,
    rot=...,
    scale=...,
    vel=...,
    mesh=my_tetmesh,          # 或 vertices=..., indices=...
    density=500.0,            # 密度
    k_mu=2.0e5,               # 剪切刚度，越大越硬
    k_lambda=2.0e5,           # 体积/压缩相关刚度，越大越难压缩
    k_damp=1e-4,              # 阻尼
    particle_radius=0.005,    # 与刚体接触的粒子半径
)
```

```
model.soft_contact_ke = 2e6
model.soft_contact_kd = 1e-7
model.soft_contact_mu = 0.5
model.shape_material_mu.fill_(1.5)
```

除此之外，这个example中的sovler架构是`one-way coupling`:sovler_vbd.py(line 250)，机械臂不会被软体的反作用力影响