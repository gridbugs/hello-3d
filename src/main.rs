extern crate cgmath;
#[macro_use]
extern crate gfx;
extern crate gfx_device_gl;
extern crate gfx_window_glutin;
extern crate glutin;
extern crate image;

use gfx::traits::FactoryExt;
use glutin::GlContext;
use gfx::{texture, Device, Factory};
use cgmath::*;

type ColourFormat = gfx::format::Srgba8;
type DepthFormat = gfx::format::DepthStencil;

gfx_vertex_struct!(Vertex {
    pos: [f32; 3] = "a_Pos",
    tex_coord: [f32; 2] = "a_TexCoord",
});

gfx_constant_struct!(Transform {
    transform: [[f32; 4]; 4] = "u_Transform",
});

gfx_constant_struct!(Properties {
    atlas_dimensions: [f32; 2] = "u_AtlasDimensions",
});

gfx_pipeline!(pipe {
    vertex_buffer: gfx::VertexBuffer<Vertex> = (),
    transform: gfx::ConstantBuffer<Transform> = "Transform",
    properties: gfx::ConstantBuffer<Properties> = "Properties",
    texture: gfx::TextureSampler<[f32; 4]> = "t_Texture",
    out_colour: gfx::BlendTarget<ColourFormat> =
        ("Target", gfx::state::ColorMask::all(), gfx::preset::blend::ALPHA),
    out_depth: gfx::DepthTarget<DepthFormat> = gfx::preset::depth::LESS_EQUAL_WRITE,
});

pub struct VerticesAndOffsets<V> {
    vertices: Vec<V>,
    offsets: Vec<gfx::VertexCount>,
}

impl<V: Clone> VerticesAndOffsets<V> {
    fn concat(&self, b: &Self) -> Self {
        let vertices = self.vertices
            .iter()
            .chain(b.vertices.iter())
            .cloned()
            .collect::<Vec<_>>();
        let offsets = self.offsets
            .iter()
            .cloned()
            .chain(b.offsets.iter().map(|i| i + self.vertices.len() as u32))
            .collect::<Vec<_>>();
        Self { vertices, offsets }
    }
}

mod wall {
    use super::*;
    const CELL_SIZE: f32 = 64.;
    const WIDTH: f32 = 16.;
    const HEIGHT: f32 = 48.;
    const FACE_TEX_OFFSET: Vector2<f32> = Vector2 { x: 384., y: 64. };
    const FACE_TEX_HEIGHT: f32 = 48.;
    const QUAD_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];

    fn extrude_disconnected<'a, I>(
        tex_start: Vector2<f32>,
        height: f32,
        vertices: I,
    ) -> VerticesAndOffsets<Vertex>
    where
        I: IntoIterator<Item = &'a (Vector2<f32>, f32)>,
    {
        let vertices = vertices
            .into_iter()
            .flat_map(|&(v, tex_x_offset)| {
                let bottom = vec3(v.x, 0., v.y);
                let top = bottom + vec3(0., height, 0.);
                let tex_top = tex_start + vec2(tex_x_offset, 0.);
                let tex_bottom = tex_top + vec2(0., height);
                vec![
                    Vertex {
                        pos: bottom.into(),
                        tex_coord: tex_bottom.into(),
                    },
                    Vertex {
                        pos: top.into(),
                        tex_coord: tex_top.into(),
                    },
                ]
            })
            .collect::<Vec<_>>();
        let offsets = (0u32..vertices.len() as u32 - 2)
            .flat_map(|i| {
                if i % 2 == 0 {
                    vec![i, i + 1, i + 2]
                } else {
                    vec![i, i + 2, i + 1]
                }
            })
            .collect::<Vec<_>>();

        VerticesAndOffsets { vertices, offsets }
    }

    pub fn straight() -> VerticesAndOffsets<Vertex> {
        const TOP_TEX_OFFSET: Vector2<f32> = Vector2 { x: 256., y: 64. };
        let start_x = CELL_SIZE / 2. - WIDTH / 2.;
        let cross_section = [
            vec2(0., start_x),
            vec2(CELL_SIZE, start_x),
            vec2(CELL_SIZE, start_x + WIDTH),
            vec2(0., start_x + WIDTH),
        ];

        let a = extrude_disconnected(
            FACE_TEX_OFFSET,
            FACE_TEX_HEIGHT,
            &[(cross_section[0], 0.), (cross_section[1], CELL_SIZE)],
        );
        let b = extrude_disconnected(
            FACE_TEX_OFFSET + vec2(CELL_SIZE, 0.),
            FACE_TEX_HEIGHT,
            &[(cross_section[2], 0.), (cross_section[3], -CELL_SIZE)],
        );
        let top = VerticesAndOffsets {
            vertices: cross_section
                .iter()
                .map(|v| {
                    let pos = vec3(v.x, HEIGHT, v.y).into();
                    let tex_coord = (TOP_TEX_OFFSET + v).into();
                    Vertex { pos, tex_coord }
                })
                .collect(),
            offsets: QUAD_INDICES.iter().cloned().collect(),
        };

        a.concat(&b).concat(&top)
    }
}

struct Camera {
    position: Vector3<f32>,
    aspect_ratio: f32,
}

impl Camera {
    fn transform(&self) -> Matrix4<f32> {
        let camera_position = self.position;
        let camera_orientation = vec3(0., -1., 0.);
        let camera_translate = Matrix4::from_translation(-camera_position);
        let camera_target_orientation = vec3(0., 0., -1.);
        let camera_rotate_axis = camera_orientation.cross(camera_target_orientation);
        let camera_rotate_angle = camera_orientation.angle(camera_target_orientation);
        let camera_rotate = Matrix4::from_axis_angle(camera_rotate_axis, camera_rotate_angle);
        let camera_perspective = cgmath::perspective(
            cgmath::Rad(::std::f32::consts::PI / 2.),
            self.aspect_ratio,
            10.,
            300.,
        );

        camera_perspective * camera_rotate * camera_translate
    }
}

fn main() {
    let (width, height) = (960., 720.);
    let mut events_loop = glutin::EventsLoop::new();
    let builder = glutin::WindowBuilder::new()
        .with_title("life-gl")
        .with_resizable(true);
    let builder = {
        let size = glutin::dpi::LogicalSize::new(width, height);
        builder
            .with_dimensions(size)
            .with_max_dimensions(size)
            .with_min_dimensions(size)
            .with_resizable(true)
    };
    let context = glutin::ContextBuilder::new().with_vsync(true);
    let (window, mut device, mut factory, rtv, dsv) =
        gfx_window_glutin::init::<ColourFormat, DepthFormat>(builder, context, &events_loop);
    let mut encoder: gfx::Encoder<_, gfx_device_gl::CommandBuffer> =
        factory.create_command_buffer().into();

    let pso = factory
        .create_pipeline_simple(
            include_bytes!("shaders/shader.150.vert"),
            include_bytes!("shaders/shader.150.frag"),
            pipe::new(),
        )
        .unwrap();

    let atlas = image::load_from_memory(include_bytes!("images/atlas.png"))
        .expect("Failed to decode test pattern")
        .to_rgba();

    let (atlas_width, atlas_height) = atlas.dimensions();
    let tex_kind = texture::Kind::D2(
        atlas_width as u16,
        atlas_height as u16,
        texture::AaMode::Single,
    );
    let tex_mipmap = texture::Mipmap::Allocated;
    let (_, texture_srv) = factory
        .create_texture_immutable_u8::<ColourFormat>(tex_kind, tex_mipmap, &[&atlas])
        .expect("Failed to create texture");

    let sampler = factory.create_sampler(texture::SamplerInfo::new(
        texture::FilterMethod::Scale,
        texture::WrapMode::Tile,
    ));

    let VerticesAndOffsets { vertices, offsets } = wall::straight();

    let wall_index_buffer = factory.create_index_buffer(&offsets[..]);
    let wall_slice = gfx::Slice {
        start: 0,
        end: offsets.len() as u32,
        base_vertex: 0,
        instances: None,
        buffer: wall_index_buffer,
    };

    let vertex_buffer = factory.create_vertex_buffer(&vertices[..]);

    let transform = factory.create_constant_buffer(1);
    let properties = factory.create_constant_buffer(1);

    let data = pipe::Data {
        vertex_buffer,
        transform,
        properties,
        texture: (texture_srv, sampler),
        out_colour: rtv,
        out_depth: dsv,
    };

    encoder.update_constant_buffer(
        &data.properties,
        &Properties {
            atlas_dimensions: [atlas_width as f32, atlas_height as f32],
        },
    );

    let walls = [vec2(0., 0.), vec2(0., 64.), vec2(64., 64.)]
        .iter()
        .map(|v| Matrix4::from_translation(vec3(v.x, 0., v.y)))
        .collect::<Vec<_>>();

    let mut camera = Camera {
        position: vec3(0., 200., 0.),
        aspect_ratio: (width / height) as f32,
    };

    let mut camera_move = vec3(0., 0., 0.);
    let mut running = true;
    while running {
        camera.position += camera_move;
        encoder.clear(&data.out_colour, [0., 0., 0., 1.]);
        encoder.clear_depth(&data.out_depth, 1.);
        let camera_transform = camera.transform();
        for t in walls.iter() {
            let final_transform = camera_transform * t;

            encoder.update_constant_buffer(
                &data.transform,
                &Transform {
                    transform: final_transform.into(),
                },
            );
            encoder.draw(&wall_slice, &pso, &data);
        }

        encoder.flush(&mut device);
        window.swap_buffers().unwrap();
        device.cleanup();

        events_loop.poll_events(|event| match event {
            glutin::Event::WindowEvent { event, .. } => match event {
                glutin::WindowEvent::CloseRequested => {
                    running = false;
                }
                glutin::WindowEvent::KeyboardInput { input, .. } => {
                    if let Some(virtual_keycode) = input.virtual_keycode {
                        let step = 1.;
                        match input.state {
                            glutin::ElementState::Pressed => match virtual_keycode {
                                glutin::VirtualKeyCode::Left => {
                                    camera_move.x = -step;
                                }
                                glutin::VirtualKeyCode::Right => {
                                    camera_move.x = step;
                                }
                                glutin::VirtualKeyCode::Up => {
                                    camera_move.z = -step;
                                }
                                glutin::VirtualKeyCode::Down => {
                                    camera_move.z = step;
                                }
                                glutin::VirtualKeyCode::A => {
                                    camera_move.y = -step;
                                }
                                glutin::VirtualKeyCode::S => {
                                    camera_move.y = step;
                                }

                                _ => (),
                            },
                            glutin::ElementState::Released => match virtual_keycode {
                                glutin::VirtualKeyCode::Left => {
                                    camera_move.x = 0.;
                                }
                                glutin::VirtualKeyCode::Right => {
                                    camera_move.x = 0.;
                                }
                                glutin::VirtualKeyCode::Up => {
                                    camera_move.z = 0.;
                                }
                                glutin::VirtualKeyCode::Down => {
                                    camera_move.z = 0.;
                                }
                                glutin::VirtualKeyCode::A => {
                                    camera_move.y = 0.;
                                }
                                glutin::VirtualKeyCode::S => {
                                    camera_move.y = 0.;
                                }
                                _ => (),
                            },
                        }
                    }
                }
                _ => (),
            },
            _ => (),
        });
    }
}
