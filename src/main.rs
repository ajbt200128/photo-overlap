use std::{time::Instant, collections::VecDeque};

use image::{GenericImage, GenericImageView, Pixel, Rgb};
use photon_rs::{
    channels::{alter_blue_channel, alter_green_channel, alter_red_channel},
    conv::noise_reduction,
    helpers::dyn_image_from_raw,
    multiple::blend,
    native::{open_image, save_image},
    transform::{crop, padding_bottom, padding_left, padding_right, padding_top, padding_uniform},
    PhotonImage, Rgba,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut img = open_image("data/01007a.jpg").expect("File should open");
    let orig = realign(img.clone(), Method::None,1000000.0,10.0);
    let new = realign(img, Method::GradientDescent,1000000.0,10.0);
    //let mut img = open_image("data/00056v.jpg").expect("File should open");
    //let orig = realign(img.clone(), Method::None,1000.0,3.0);
    //let new = realign(img, Method::GradientDescent,1000.0,3.0);
    save_image(orig, "img.jpg")?;
    save_image(new, "img2.jpg")?;
    Ok(())
}
enum Method {
    Scale,
    GradientDescent,
    None,
}

fn realign(mut img: PhotonImage, method: Method,delta:f64,epsilon:f64) -> PhotonImage {
    let width = img.get_width();
    let height = img.get_height();
    let b_channel = crop(&mut img, 0, 0, width, height / 3);
    let g_channel = crop(&mut img, 0, height / 3, width, 2 * height / 3);
    let r_channel = crop(&mut img, 0, 2 * height / 3, width, 3 * height / 3);
    let (mut r, mut g, mut b) = match method {
        Method::Scale => todo!(),
        Method::GradientDescent => {
            let (r, g) = gradient_descent(r_channel, g_channel, delta, epsilon);
            let (r, g) = gradient_descent(r, g, delta/2.0, epsilon/3.0);
            let (g, b) = gradient_descent(g, b_channel, delta, epsilon);
            let (g, b) = gradient_descent(g, b, delta/2.0, epsilon/3.0);
            let (r, g) = gradient_descent(r, g, delta, epsilon);
            let (r, g) = gradient_descent(r, g, delta/2.0, epsilon/3.0);
            (r, g, b)
        }
        Method::None => (r_channel, g_channel, b_channel),
    };

    reverse_grayscale(&mut r, 0);
    reverse_grayscale(&mut g, 1);
    reverse_grayscale(&mut b, 2);
    let orig = overlay(r, b);
    save_image(orig.clone(), "inter.jpg").unwrap();
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
            //////println!("{:?} {:?}", a.2.channels(),b.2.channels());

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
    let b_x_pad = pi_a.get_width() - pi_b.get_width();
    if pi_a.get_width() < pi_b.get_width() {
        std::mem::swap(&mut pi_a, &mut pi_b);
    }
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

fn difference(pi_a: &PhotonImage, pi_b: &PhotonImage) -> f64 {
    let start = Instant::now();
    let pixels_a = dyn_image_from_raw(pi_a);
    let pixels_a = pixels_a.pixels();
    let pixels_b = dyn_image_from_raw(pi_b);
    let pixels_b = pixels_b.pixels();
    let end = Instant::now();
    //let sum_sqd_diff:u128 = pixels_a.clone().into_iter().zip(pixels_b.clone().into_iter()).map(|((_,_,a),(_,_,b))| (a.channels()[0] as f64-b.channels()[0] as f64).powf(2.0) as u128).sum();
    let start = Instant::now();
    let (a, b, c) = pixels_a
        .into_iter()
        .zip(pixels_b.into_iter())
        .map(|((_, _, a), (_, _, b))| {
            let a = a.channels()[0] as f64;
            let b = b.channels()[0] as f64;
            (a * b, a.powf(2.0), b.powf(2.0))
        })
        .fold((0.0, 0.0, 0.0), |(x, y, z), (a, b, c)| {
            (x + a, b + y, z + c)
        });
    let ncc = a / (b.sqrt() * c.sqrt());
    let end = Instant::now();
    ////println!("time d: {:?}", end.duration_since(start));

    //////println!("ssd {}",sum_sqd_diff);
    ////println!("ncc {}", ncc);
    return ncc;
}

fn pad_photo(
    (x, y): (f64, f64),
    pi_a: &PhotonImage,
    pi_b: &PhotonImage,
) -> (PhotonImage, PhotonImage) {
    let (pi_a_x, pi_b_x) = if x > 0.0 {
        (x.round().abs(), 0.0)
    } else {
        (0.0, x.round().abs())
    };
    let (pi_a_y, pi_b_y) = if y > 0.0 {
        (y.round().abs(), 0.0)
    } else {
        (0.0, y.round().abs())
    };
    let pi_a_pad = padding_left(&pi_a, pi_a_x as u32, Rgba::new(0, 0, 0, 0));
    let pi_a_pad = padding_top(&pi_a, pi_a_y as u32, Rgba::new(0, 0, 0, 0));
    let pi_b_pad = padding_left(&pi_b, pi_b_x as u32, Rgba::new(0, 0, 0, 0));
    let pi_b_pad = padding_top(&pi_b, pi_b_y as u32, Rgba::new(0, 0, 0, 0));
    (pi_a_pad, pi_b_pad)
}

fn gradient_descent(
    pi_a: PhotonImage,
    pi_b: PhotonImage,
    delta: f64,
    epsilon: f64,
) -> (PhotonImage, PhotonImage) {
    let mut padding_x = 0.0;
    let mut padding_y = 0.0;
    let mut seen_x:VecDeque<i32> = VecDeque::new();
    let mut seen_y:VecDeque<i32> = VecDeque::new();
    let (mut seen_x_cnt,mut seen_y_cnt) = (0,0);
    let mut delta_x = delta;
    let mut delta_y = delta;
    let mut epsilon_x = epsilon;
    let mut epsilon_y = epsilon;
    let mut rng = rand::thread_rng();
    let mut pi_a_pad_final = pi_a.clone();
    let mut pi_b_pad_final = pi_b.clone();
    for _ in 1..300 {
        let start = Instant::now();
        let (pi_a_pad, pi_b_pad) = pad_photo((padding_x, padding_y), &pi_a, &pi_b);
        let (pi_a_pad_x, pi_b_pad_x) = pad_photo((padding_x + epsilon_x, padding_y), &pi_a, &pi_b);
        let (pi_a_pad_y, pi_b_pad_y) = pad_photo((padding_x, padding_y + epsilon_y), &pi_a, &pi_b);
        let end = Instant::now();

        let start = Instant::now();
        let diff = 1.0 - difference(&pi_a_pad, &pi_b_pad);
        let diff_x_p = 1.0 - difference(&pi_a_pad_x, &pi_b_pad_x);
        let diff_y_p = 1.0 - difference(&pi_a_pad_y, &pi_b_pad_y);
        let end = Instant::now();
        let mut gradient_x =  delta_x * ((diff - diff_x_p) / epsilon_x);
        let mut gradient_y = delta_y * (diff - diff_y_p) / epsilon_y;
        println!("G_Y {}",gradient_y);
      //if gradient_x.abs() != 0.0{
      //    padding_x -= gradient_x.abs().log2() * sign_x;
      //}
      //if gradient_y.abs() != 0.0{
      //    padding_y -= gradient_y.abs().log2() * sign_y;
      //}
        let sign_x = gradient_x / gradient_x.abs();
        let sign_y = gradient_y / gradient_y.abs();

        if gradient_x.abs() >= 1.0{
            gradient_x = gradient_x.abs().log2() * sign_x
        }
        if gradient_y.abs() >= 1.0{
            gradient_y = gradient_y.abs().log2() * sign_y;
        }
        gradient_x = (gradient_x * 100.0).trunc() / 100.0;
        gradient_y = (gradient_y * 100.0).trunc() / 100.0;
        if seen_x.contains(&(gradient_x as i32)){
            seen_x_cnt+=1;
        }else{
            seen_x_cnt = 0;
        }
        if seen_y.contains(&(gradient_y as i32)){
            seen_y_cnt+=1;
        }else{
            seen_y_cnt = 0;
        }
        seen_x.push_back(gradient_x as i32);
        seen_y.push_back(gradient_y as i32);
        if seen_x.len() > 10{
            seen_x.pop_front();
        }
        if seen_y.len() > 10{
            seen_y.pop_front();
        }
        padding_x += gradient_x;
        padding_y += gradient_y;
        println!("x: {} {:.5} {:.5} {:.5} {} {:?}", gradient_x, padding_x, diff, diff_x_p,delta_x,epsilon_x);
        println!("y: {} {:.5} {:.5} {:.5} {} {:?}", gradient_y, padding_y, diff, diff_y_p,delta_y,epsilon_y);
      //if gradient_x.abs() < epsilon_x && epsilon_x >= 1.1 {
      //    epsilon_x /= 1.1;
      //}
      //if gradient_y.abs() < epsilon_y && epsilon_y >= 1.1 {
      //    epsilon_y /= 1.1;
      //}
        if seen_x_cnt > 4{
            delta_x /= 10.0;
           // println!("A--------------");
            //return (pi_a_pad, pi_b_pad);
        }
        if seen_y_cnt > 4{
            delta_y /= 10.0;
            // println!("A--------------");
            //return (pi_a_pad, pi_b_pad);
        }
        if gradient_x.abs() < 0.01 && gradient_y.abs() < 0.01{
            println!("B--------------");
            return (pi_a_pad, pi_b_pad);
        }
        pi_a_pad_final = pi_a_pad;
        pi_b_pad_final = pi_b_pad;
    }
    println!("C--------------");
    (pi_a_pad_final, pi_b_pad_final)
}
