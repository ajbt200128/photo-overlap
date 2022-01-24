use std::{collections::VecDeque, time::Instant};

use cached::proc_macro::cached;
use image::{GenericImage, GenericImageView, Pixel, Rgb};
use photon_rs::{
    channels::{self, alter_blue_channel, alter_green_channel, alter_red_channel},
    conv::noise_reduction,
    helpers::dyn_image_from_raw,
    multiple::blend,
    native::{open_image, save_image},
    transform::{
        crop, padding_bottom, padding_left, padding_right, padding_top, padding_uniform, resize,
        SamplingFilter,
    },
    PhotonImage, Rgba,
};
use clap::Parser;

#[derive(Parser,Debug)]
#[clap(author, version, about, long_about = None)]
struct Args{
    file:String,
    learn:f64,
    epsilon:f64,
    scale:u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let img = open_image(&args.file).expect("File should open");
    //let img = open_image("data/00056v.jpg").expect("File should open");
    println!("w,h {} {}", img.get_width(), img.get_height());
    let orig = realign(img.clone(), Method::None, 10000.0, 1.0, 1);
    //let brute = realign(img.clone(), Method::Scale, 10000.0, 100.0, 1);
    let new = realign(img, Method::GradientDescent, args.learn, args.epsilon, args.scale);
    save_image(orig, "img.jpg")?;
    save_image(new.clone(), "img2.jpg")?;
    save_image(new, format!("processed/{}",args.file).as_str())?;
    //save_image(brute, "img3.jpg")?;
    Ok(())
}
enum Method {
    Scale,
    GradientDescent,
    None,
}

#[derive(Clone, Copy)]
enum Channel {
    R = 0,
    G = 1,
    B = 2,
}
#[derive(Clone, Copy)]
struct Channels {
    a: Channel,
    b: Channel,
}

impl Channels {
    fn new((a, b): (Channel, Channel)) -> Self {
        Self { a, b }
    }
}
impl From<(Channel, Channel)> for Channels {
    fn from(p: (Channel, Channel)) -> Self {
        Channels::new(p)
    }
}

fn realign(img: PhotonImage, method: Method, delta: f64, epsilon: f64, scale: u32) -> PhotonImage {
    let mut img_orig = img.clone();
    let mut img = resize(
        &img,
        img.get_width() / scale,
        img.get_height() / scale,
        SamplingFilter::Nearest,
    );
    let width = img.get_width();
    let height = img.get_height();
    let b_channel = crop(&mut img, 0, 0, width, height / 3);
    let g_channel = crop(&mut img, 0, height / 3, width, 2 * height / 3);
    let r_channel = crop(&mut img, 0, 2 * height / 3, width, 3 * height / 3);
    let (p_rg, p_gb,p_rb,p_rg_p, p_gb_p,p_rb_p) = match method {
        Method::Scale => {
            let (r, g, p_rg) =
                brute_force(r_channel, g_channel, (Channel::R, Channel::G).into(), 15);
            let (g, b, p_gb) = brute_force(g, b_channel, (Channel::G, Channel::B).into(), 15);
            let (_, _, p_rb) = brute_force(r, b, (Channel::R, Channel::B).into(), 15);
            (p_rg, p_gb,p_rb,0.0,0.0,0.0)
        }
        Method::GradientDescent => {
            let (g, b, p_gb) = gradient_descent(
                g_channel,
                b_channel,
                (Channel::G, Channel::B).into(),
                delta,
                epsilon,
                true,
            );

            let (g, b, p_gb_p) = gradient_descent(
                g,
                b,
                (Channel::G, Channel::B).into(),
                delta,
                epsilon,
                false,
            );
            let (r, g, p_rg) = gradient_descent(
                r_channel,
                g,
                (Channel::R, Channel::G).into(),
                delta,
                epsilon,
                true
            );
            let (r, g, p_rg_p) =
                gradient_descent(r, g, (Channel::R, Channel::G).into(), delta, epsilon,false);

            let (r, b, p_rb) =
                gradient_descent(r, b, (Channel::R, Channel::B).into(), delta, epsilon,true);
            let (r, b, p_rb_p) =
                gradient_descent(r, b, (Channel::R, Channel::B).into(), delta, epsilon,false);
            (p_rg, p_gb,p_rb,p_rg_p, p_gb_p,p_rb_p)
        }
        Method::None => (0.0, 0.0,0.0,0.0, 0.0,0.0),
    };

    let width = img_orig.get_width();
    let height = img_orig.get_height();
    let p_rg = (p_rg * scale as f32) as i32;
    let p_gb = (p_gb * scale as f32) as i32;
    let p_rb = (p_rb * scale as f32) as i32;
    let p_rg_p = (-p_rg_p * scale as f32) as i32;
    let p_gb_p = (-p_gb_p * scale as f32) as i32;
    let p_rb_p = (-p_rb_p * scale as f32) as i32;
    let b = crop(&mut img_orig, 0, 0, width, height / 3);
    let g = crop(&mut img_orig, 0, height / 3, width, 2 * height / 3);
    let r = crop(&mut img_orig, 0, 2 * height / 3, width, 3 * height / 3);
    let (mut g, mut b) = pad_photo(p_gb_p, &g, &b,false);
    let (mut r, mut b) = pad_photo(p_rb, &r, &b,true);
    let (r, g) = pad_photo(p_rg, &r, &g,true);
    let (mut r, g) = pad_photo(p_rg_p, &r, &g,false);
    let (mut g, b) = pad_photo(p_gb, &g, &b,true);
    let (mut r, mut b) = pad_photo(p_rb_p, &r, &b,false);
    println!("rg [{},{}] rb [{},{}] gb [{},{}]",p_rg,&p_rg_p,&p_rb,p_rb_p,p_gb,p_gb_p);
    reverse_grayscale(&mut r, 0);
    reverse_grayscale(&mut g, 1);
    reverse_grayscale(&mut b, 2);
    let orig = overlay(r, b);
    overlay(orig, g)
}

fn component_overlay(pi_a: &PhotonImage, pi_b: &PhotonImage) -> PhotonImage {
    let mut img_a = dyn_image_from_raw(pi_a);
    let img_b = dyn_image_from_raw(pi_b);
    let img_a_pixels = img_a.clone();
    let pixels_a = img_a_pixels.pixels();
    let img_b_pixels = img_b;
    let pixels_b = img_b_pixels.pixels();
    pixels_a
        .into_iter()
        .zip(pixels_b.into_iter())
        .for_each(|(mut a, b)| {
            a.2.channels_mut()[0] =
                (a.2.channels_mut()[0] as u16 + b.2.channels()[0] as u16).clamp(0, 255) as u8;
            a.2.channels_mut()[1] =
                (a.2.channels_mut()[1] as u16 + b.2.channels()[1] as u16).clamp(0, 255) as u8;
            a.2.channels_mut()[2] =
                (a.2.channels_mut()[2] as u16 + b.2.channels()[2] as u16).clamp(0, 255) as u8;
            img_a.put_pixel(a.0, a.1, a.2);
        });
    let raw_pixels = img_a.to_bytes();
    PhotonImage::new(raw_pixels, pi_a.get_width(), pi_a.get_height())
}

fn overlay(mut pi_a: PhotonImage, mut pi_b: PhotonImage) -> PhotonImage {
    if pi_a.get_width() < pi_b.get_width() {
        std::mem::swap(&mut pi_a, &mut pi_b);
    }
    let b_x_pad = pi_a.get_width() - pi_b.get_width();
    let mut pi_b = padding_right(&pi_b, b_x_pad, Rgba::new(0, 0, 0, 0));

    if pi_a.get_height() < pi_b.get_height() {
        std::mem::swap(&mut pi_a, &mut pi_b);
    }
    let b_y_pad = pi_a.get_height() - pi_b.get_height();
    let pi_b = padding_bottom(&pi_b, b_y_pad, Rgba::new(0, 0, 0, 0));

    component_overlay(&pi_a, &pi_b)
}

fn reverse_grayscale(photon_image: &mut PhotonImage, channel: usize) {
    if channel != 0 {
        alter_red_channel(photon_image, -255);
    }
    if channel != 1 {
        alter_green_channel(photon_image, -255);
    }
    if channel != 2 {
        alter_blue_channel(photon_image, -255);
    }
}

// Should DP
fn difference(pi_a: &PhotonImage, pi_b: &PhotonImage) -> f64 {
    let pixels_a = dyn_image_from_raw(pi_a);
    let pixels_a = pixels_a.pixels();
    let pixels_b = dyn_image_from_raw(pi_b);
    let pixels_b = pixels_b.pixels();
    let (a, b, c) = pixels_a
        .into_iter()
        .zip(pixels_b.into_iter())
        .map(|((_, _, a), (_, _, b))| {
            let b = b.to_luma();
            let a = a.to_luma();
            let mut b = b.channels()[0] as f64;
          //if a.channels()[3] == 0 {
          //    b = 0.0;
          //}
            let a = a.channels()[0] as f64;
            (a * b, a.powf(2.0), b.powf(2.0))
        })
        .fold((0.0, 0.0, 0.0), |(x, y, z), (a, b, c)| {
            (x + a, b + y, z + c)
        });
    a / (b.sqrt() * c.sqrt())
}

fn pad_photo(x: i32, pi_a: &PhotonImage, pi_b: &PhotonImage,side:bool) -> (PhotonImage, PhotonImage) {
    let (pi_a_y, pi_b_y) = if x > 0 { (x.abs(), 0) } else { (0, x.abs()) };
    if side{
        let pi_a_pad = padding_top(&pi_a, pi_a_y as u32, Rgba::new(255, 255, 255, 255));
        let pi_b_pad = padding_top(&pi_b, pi_b_y as u32, Rgba::new(255, 255, 255, 255));
        (pi_a_pad, pi_b_pad)
    }else{
        let pi_a_pad = padding_left(&pi_a, pi_a_y as u32, Rgba::new(255, 255, 255, 255));
        let pi_b_pad = padding_left(&pi_b, pi_b_y as u32, Rgba::new(255, 255, 255, 255));
        (pi_a_pad, pi_b_pad)
    }
}

#[cached(
    key = "(i32,u8,u8,bool)",
    convert = r#"{(y,channels.a as u8,channels.b as u8,side)}"#
)]
fn pad_and_diff(
    y: i32,
    channels: Channels,
    pi_a: &PhotonImage,
    pi_b: &PhotonImage,
    side:bool,
) -> (PhotonImage, PhotonImage, f64) {
    let (pi_a_pad, pi_b_pad) = pad_photo(y, pi_a, pi_b,true);
    let diff = difference(&pi_a_pad, &pi_b_pad);
    (pi_a_pad, pi_b_pad, diff)
}

fn brute_force(
    pi_a: PhotonImage,
    pi_b: PhotonImage,
    channels: Channels,
    search_radius: i32,
) -> (PhotonImage, PhotonImage, f32) {
    let mut best = 0.0;
    let mut best_a = pi_a.clone();
    let mut best_b = pi_b.clone();
    let mut best_p = 0.0;
    for y in (-search_radius)..search_radius {
        println!("{}", y);
        let (pi_a_pad, pi_b_pad, diff) = pad_and_diff(y, channels, &pi_a, &pi_b,true);
        if diff > best {
            println!("{} {}", best, diff);
            best = diff;
            best_a = pi_a_pad;
            best_b = pi_b_pad;
            best_p = y as f32;
        }
    }
    (best_a, best_b, best_p)
}

fn gradient_descent(
    pi_a: PhotonImage,
    pi_b: PhotonImage,
    channels: Channels,
    delta: f64,
    mut epsilon: f64,
    side:bool
) -> (PhotonImage, PhotonImage, f32) {
    let mut padding = 0.0;
    let mut seen: VecDeque<i32> = VecDeque::new();
    let mut seen_cnt = 0;
    let mut delta = delta;
    let mut best = 1.0;
    let mut best_pad = 0.0;
    let mut pi_a_pad_final = pi_a.clone();
    let mut pi_b_pad_final = pi_b.clone();
    for _ in 1..300 {
        let (pi_a_pad, pi_b_pad, diff) = pad_and_diff((padding) as i32, channels, &pi_a, &pi_b,side);
        let (_, _, diff_p) = pad_and_diff((padding + epsilon) as i32, channels, &pi_a, &pi_b,side);

        let diff = 1.0 - diff;
        let diff_p = 1.0 - diff_p;
        let mut gradient = delta * ((diff - diff_p) / epsilon);
        let sign = gradient / gradient.abs();

        if gradient.abs() >= 1.0 {
            gradient = gradient.abs().log2() * sign;
        }
        gradient = (gradient * 100.0).trunc() / 100.0;
        if seen.contains(&(gradient as i32)) {
            seen_cnt += 1;
        } else {
            seen_cnt = 0;
        }
        seen.push_back(gradient as i32);
        if seen.len() > 10 {
            seen.pop_front();
        }
        println!(
            "x: {} {:.5} {:.5} {:.5} {} {:?}",
            gradient, padding, diff, diff_p, delta, epsilon
        );
        padding += gradient;
        if seen_cnt > 4 {
            delta /= 10.0;
        }
        if epsilon > gradient {
            epsilon/=1.1;
        }
        if diff < best {
            best = diff;
            pi_a_pad_final = pi_a_pad;
            pi_b_pad_final = pi_b_pad;
            best_pad = padding
        }
        if gradient.abs() < 0.01 {
            println!("-------");
            return (pi_a_pad_final, pi_b_pad_final, best_pad as f32);
        }
    }
    (pi_a_pad_final, pi_b_pad_final, best_pad as f32)
}
