extern crate cgmath;
#[macro_use]
extern crate gfx;
extern crate gfx_device_gl;
extern crate gfx_window_glutin;
extern crate glutin;

use gfx::traits::FactoryExt;
use glutin::GlContext;
use gfx::Device;
use cgmath::*;

type ColourFormat = gfx::format::Srgba8;
type DepthFormat = gfx::format::DepthStencil;

gfx_vertex_struct!(Vertex {
    pos: [f32; 3] = "a_Pos",
});

gfx_constant_struct!(Transform {
    transform: [[f32; 4]; 4] = "u_Transform",
});

gfx_pipeline!(pipe {
    vertex_buffer: gfx::VertexBuffer<Vertex> = (),
    transform: gfx::ConstantBuffer<Transform> = "Transform",
    out_colour: gfx::BlendTarget<ColourFormat> =
        ("Target", gfx::state::ColorMask::all(), gfx::preset::blend::ALPHA),
    out_depth: gfx::DepthTarget<DepthFormat> = gfx::preset::depth::LESS_EQUAL_WRITE,
});

struct VerticesAndOffsets<V> {
    vertices: Vec<V>,
    offsets: Vec<gfx::VertexCount>,
}

struct Wall {
    start: Vector2<f32>,
    end: Vector2<f32>,
    base: f32,
    height: f32,
}

impl Wall {
    fn buffers() -> VerticesAndOffsets<Vertex> {
        let vertices = [
            vec3(0., 0., 0.),
            vec3(0., 1., 0.),
            vec3(1., 1., 0.),
            vec3(1., 0., 0.),
        ].iter()
            .map(|&pos| Vertex { pos: pos.into() })
            .collect::<Vec<_>>();
        let offsets = vec![0, 1, 2, 0, 2, 3];
        VerticesAndOffsets { vertices, offsets }
    }
    fn normalize(&self) -> Matrix4<f32> {
        let start_to_end = self.end - self.start;
        let base_direction = vec2(1., 0.);
        let scale = Matrix4::from_nonuniform_scale(start_to_end.magnitude(), self.height, 1.);
        let rotate = Matrix4::from_angle_y(start_to_end.angle(base_direction));
        let translate = Matrix4::from_translation(vec3(self.start.x, self.base, self.start.y));
        translate * rotate * scale
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
            80.,
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

    let VerticesAndOffsets { vertices, offsets } = Wall::buffers();

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

    let data = pipe::Data {
        vertex_buffer,
        transform,
        out_colour: rtv,
        out_depth: dsv,
    };

    let walls = [
        Wall {
            start: vec2(0., 0.),
            end: vec2(0., 10.),
            base: 0.,
            height: 10.,
        },
        Wall {
            start: vec2(10., 5.),
            end: vec2(10., 25.),
            base: 0.,
            height: 5.,
        },
        Wall {
            start: vec2(-5., -5.),
            end: vec2(-15., 5.),
            base: 0.,
            height: 10.,
        },
    ].iter()
        .map(|w| w.normalize())
        .collect::<Vec<_>>();

    let mut camera = Camera {
        position: vec3(0., 40., 0.),
        aspect_ratio: (width / height) as f32,
    };

    let mut running = true;
    while running {
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
                        let step = 0.2;
                        match input.state {
                            glutin::ElementState::Pressed => match virtual_keycode {
                                glutin::VirtualKeyCode::Left => {
                                    camera.position.x -= step;
                                }
                                glutin::VirtualKeyCode::Right => {
                                    camera.position.x += step;
                                }
                                glutin::VirtualKeyCode::Up => {
                                    camera.position.z -= step;
                                }
                                glutin::VirtualKeyCode::Down => {
                                    camera.position.z += step;
                                }
                                _ => (),
                            },
                            _ => (),
                        }
                    }
                }
                _ => (),
            },
            _ => (),
        });
    }
}
