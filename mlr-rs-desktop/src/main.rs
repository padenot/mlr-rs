#[macro_use]
extern crate log;
extern crate mlr_rs;
extern crate audio_clock;
extern crate cubeb;
extern crate pretty_env_logger;
use std::thread;
use std::time::Duration;

use mlr_rs::MLR;
use audio_clock::*;

fn main() {
    pretty_env_logger::init();

    let (mut clock_updater, clock_receiver) = audio_clock(128., 48000);

    let (mut mlr, mut renderer) = MLR::new(clock_receiver);

    // set up audio output
    let ctx = cubeb::init("mlr-rs").expect("Failed to create cubeb context");
    let params = cubeb::StreamParamsBuilder::new()
        .format(cubeb::SampleFormat::Float32NE)
        .rate(48000)
        .channels(1)
        .layout(cubeb::ChannelLayout::MONO)
        .take();

    let mut builder = cubeb::StreamBuilder::new();
    builder
        .name("mlr-rs")
        .default_output(&params)
        .latency(256)
        .data_callback(move |_input: &[f32], output| {
            renderer.render(output);
            clock_updater.increment(output.len());
            output.len() as isize
        })
        .state_callback(|state| {
            info!("stream {:?}", state);
        });

    let stream = builder.init(&ctx).expect("Failed to create cubeb stream");

    stream.start().unwrap();

    loop {
        mlr.main_thread_work();
        mlr.poll_input();
        mlr.render();
        thread::sleep(Duration::from_millis(1));
    }
}
