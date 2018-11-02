extern crate log;
extern crate mlr_rs;
extern crate audio_clock;
extern crate bela;
extern crate pretty_env_logger;
use std::thread;
use std::time::Duration;

use bela::*;
use mlr_rs::{MLR, MLRRenderer};
use audio_clock::*;

fn go() -> Result<bool, bela::error::Error> {
    let (mut clock_updater, clock_receiver) = audio_clock(128., 44100);

    let (mut mlr, renderer) = MLR::new(clock_receiver);

    let mut setup = |_context: &mut Context, _user_data: &mut MLRRenderer| -> Result<(), error::Error> {
        println!("Setting up");
        Ok(())
    };

    let mut cleanup = |_context: &mut Context, _user_data: &mut MLRRenderer| {
        println!("Cleaning up");
    };

    let mut render = |context: &mut Context, renderer: &mut MLRRenderer| {
        let frames = context.audio_frames();
        let rate = context.audio_sample_rate();
        let output = context.audio_out();
        let mut out: [f32; 16] = [0.0; 16];
        renderer.render(&mut out);
        for i in 0..16 {
            output[i * 2] = out[i];
            output[i * 2 + 1] = out[i];
        }
        clock_updater.increment(frames * 2);
    };

    let user_data = AppData::new(renderer, &mut render, Some(&mut setup), Some(&mut cleanup));
    let mut bela_app = Bela::new(user_data);
    let mut settings = InitSettings::default();
    bela_app.init_audio(&mut settings)?;
    bela_app.start_audio()?;

    loop {
        mlr.main_thread_work();
        mlr.poll_input();
        mlr.render();
        thread::sleep(Duration::from_millis(16));
    }

    bela_app.stop_audio();
    bela_app.cleanup_audio();
}

fn main() {
    pretty_env_logger::init();

    match go() {
        Ok(_) => { println!("??"); }
        Err(_) => { println!("!!"); }
    }
}
