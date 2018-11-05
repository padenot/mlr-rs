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

struct MonomeTask<F> {
    callback: F,
    args: MLR
}

impl<F> Auxiliary for MonomeTask<F>
where F: FnMut(&mut MLR),
      for<'r> F: FnMut(&'r mut MLR)
{
    type Args = MLR;

    fn destructure(&mut self) -> (&mut FnMut(&mut MLR), &mut Self::Args) {
        let MonomeTask {
            callback,
            args,
        } = self;

        (callback, args)
    }
}

type BelaApp<'a> = Bela<AppData<'a, MLRRenderer>>;

fn go() -> Result<(), bela::error::Error> {
    let (mut clock_updater, clock_receiver) = audio_clock(128., 44100);
    println!("loading samples & decoding...");

    let (mlr, renderer) = MLR::new(clock_receiver);

    println!("ok!");

    let mut monome_task = MonomeTask {
        callback: |mlr: &mut MLR| {
            loop {
                mlr.main_thread_work();
                mlr.poll_input();
                mlr.render();
                thread::sleep(Duration::from_millis(16));
            }
        },
        args: mlr
    };

    let mut setup = |_context: &mut Context, _user_data: &mut MLRRenderer| -> Result<(), error::Error> {
        println!("Setting up");
        let task = BelaApp::create_auxiliary_task(&mut monome_task, 10, "monome_task");
        BelaApp::schedule_auxiliary_task(&task)?;
        println!("ok");
        Ok(())
    };

    let mut cleanup = |_context: &mut Context, _user_data: &mut MLRRenderer| {
        println!("Cleaning up");
    };

    let mut render = |context: &mut Context, renderer: &mut MLRRenderer| {
        let frames = context.audio_frames();
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

    let mut settings = InitSettings::default();

    Bela::new(user_data).run(&mut settings)
}

fn main() {
    pretty_env_logger::init();

    match go() {
        Ok(_) => { println!("??"); }
        Err(_) => { println!("!!"); }
    }
}
