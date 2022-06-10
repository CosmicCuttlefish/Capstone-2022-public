use arrayfire::*;

const NUM_AGENTS: u64 = 1_000_000;
const HEIGHT: u64 = 1920;
const WIDTH: u64 = 1920;
const MARGIN: f32 = 10.0;
const SPEED: f32 = 0.50;
const THETA: f32 = 1.9548;

const SENSOR_DIST: f32 = 20.0;
const TURN_SPEED: f32 = 0.0524;
const DECAY_RATE: f32 = 0.75;
const DIFFUSE_RATE: f32 = 0.30;
const MINUS_PI: f32 = -3.1415;
const PI: f32 = 3.1415;

fn main() {
    set_device(0);
    info();
    slime_sim();
}

fn slime_sim() {
    let big_dims = dim4!(HEIGHT, WIDTH);
    let val_table = constant::<f32>(2.0, dim4!(NUM_AGENTS, 1));
    let mut agent_map = constant::<f32>(HEIGHT as f32 / 2.0, Dim4::new(&[NUM_AGENTS, 3, 1, 1]));

    let mut trail_map = constant::<f32>(0.0, big_dims);
    let mut angle_table = randu::<f32>(dim4!(NUM_AGENTS));
    angle_table = angle_table * 6.28 as f32;

    set_col(&mut agent_map, &angle_table, 2);
    let margin = constant::<f32>(MARGIN, dim4!(1));

    let height = constant::<f32>(HEIGHT as f32, dim4!(NUM_AGENTS));
    let width = constant::<f32>(WIDTH as f32, dim4!(NUM_AGENTS));
    set_col(&mut agent_map, &(&width / 2 as f32), 1);

    let gaus = gaussian_kernel(3, 3, DIFFUSE_RATE.into(), DIFFUSE_RATE.into());

    let mut counter = 0;

    let win = Window::new(1920, 1920, "P. Polycephalum".to_string());
    while counter < 100000 {
        let mut rand_buffer = randu::<f32>(dim4!(NUM_AGENTS));
        rand_buffer = normalise(&rand_buffer);
        let seed_setter = counter as u64;
        set_seed(seed_setter);

        let ty = sin(&col(&agent_map, 2));
        let tx = cos(&col(&agent_map, 2));
        let mut cy = &ty * SPEED;
        let mut cx = &tx * SPEED;
        cy = &cy + &col(&agent_map, 1);
        cx = &cx + &col(&agent_map, 0);

        let cx = clamp::<f32, f32>(&cx, &MARGIN, &(&(WIDTH as f32) - &MARGIN), true);
        let cy = clamp::<f32, f32>(&cy, &MARGIN, &(&(HEIGHT as f32) - &MARGIN), true);

        let flat_trails = flat(&trail_map);
        let leftsense = col(&agent_map, 2) + THETA;
        let csense = col(&agent_map, 2);
        let rightsense = col(&agent_map, 2) - THETA;

        let leftpointy = (sin(&leftsense) * SENSOR_DIST) + col(&agent_map, 1);
        let leftpointx = (cos(&leftsense) * SENSOR_DIST) + col(&agent_map, 0);

        let centerpointy = (sin(&csense) * SENSOR_DIST) + col(&agent_map, 1);
        let centerpointx = (cos(&csense) * SENSOR_DIST) + col(&agent_map, 0);

        let rightpointy = (sin(&rightsense) * SENSOR_DIST) + col(&agent_map, 1);
        let rightpointx = (cos(&rightsense) * SENSOR_DIST) + col(&agent_map, 0);

        let left_flat = ix(&leftpointx, &leftpointy);
        let left_xy = view!(flat_trails[left_flat]);

        let right_flat = ix(&rightpointx, &rightpointy);
        let right_xy = view!(flat_trails[right_flat]);

        let center_flat = ix(&centerpointx, &centerpointy);
        let center_xy = view!(flat_trails[center_flat]);

        let left_gt_right = gt(&left_xy, &right_xy, true);
        let left_gt_center = gt(&left_xy, &center_xy, true);
        let right_gt_center = gt(&right_xy, &center_xy, true);
        let right_gt_left = gt(&right_xy, &left_xy, true);
        let center_gt_left = gt(&center_xy, &left_xy, true);
        let center_gt_right = gt(&center_xy, &right_xy, true);

        let mut weigh_forward = and(&center_gt_left, &center_gt_right, true);
        let weigh_left = and(&left_gt_center, &left_gt_right, true);
        let weigh_right = and(&right_gt_center, &right_gt_left, true);
        weigh_forward = bitnot(&or(&weigh_left, &weigh_right, true)) + &weigh_forward;
        let turn_weight = (&weigh_left.cast::<f32>() * -1 * TURN_SPEED * &rand_buffer)
            + (&weigh_right.cast::<f32>() * 1 * TURN_SPEED * &rand_buffer)
            + (&weigh_forward.cast::<f32>() * TURN_SPEED * 2.0 * (&rand_buffer - 0.5));

        let chase_weight = &turn_weight.cast::<f32>() * PI;

        let cyo = ge(&cy, &(&height - &margin), true);
        let cxo = ge(&cx, &(&width - &margin), true);
        let ayo = le(&cy, &margin, true);
        let axo = le(&cx, &margin, true);
        let mut bounds_angle =
            &cxo.cast::<f32>() + &cyo.cast::<f32>() + &axo.cast::<f32>() + &ayo.cast::<f32>();
        bounds_angle = &bounds_angle * MINUS_PI;
        bounds_angle = &col(&agent_map, 2) + chase_weight + bounds_angle;
        set_col(&mut agent_map, &bounds_angle, 2);

        set_col(&mut agent_map, &cx, 0);
        set_col(&mut agent_map, &cy, 1);
        let add_map = sparse(
            HEIGHT,
            WIDTH,
            &val_table,
            &cx.cast::<i32>(),
            &cy.cast::<i32>(),
            SparseFormat::COO,
        );

        let holder = sparse_convert_to(&add_map, SparseFormat::CSR);
        let add_map_d = sparse_to_dense(&holder);
        counter += 1;

        trail_map = &trail_map + add_map_d;
        trail_map = &trail_map - DECAY_RATE as f32;
        trail_map = convolve2(&trail_map, &gaus, ConvMode::DEFAULT, ConvDomain::SPATIAL);
        trail_map = clamp::<f32, f32>(&trail_map, &0.0, &1.0, true);

        win.draw_image(&trail_map, None);
    }
}

fn normalise(a: &Array<f32>) -> Array<f32> {
    a / (max_all(&abs(a)).0 as f32)
}

fn ix(x: &Array<f32>, y: &Array<f32>) -> Array<f32> {
    (x * WIDTH as f32) + y
}
