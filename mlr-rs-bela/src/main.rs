extern crate log;
extern crate mlr_rs;
extern crate bela;
extern crate pretty_env_logger;
extern crate mbms_traits;
extern crate monome;

use std::thread;
use std::time::Duration;

use bela::*;
use mlr_rs::{MLR, MLRRenderer};
use mbms_traits::*;
use monome::*;

struct Control {
    mlr: MLR,
    monome: Monome
}

struct MonomeTask<F> {
    callback: F,
    args: Control
}

impl<F> Auxiliary for MonomeTask<F>
where F: FnMut(&mut Control),
      for<'r> F: FnMut(&'r mut Control)
{
    type Args = Control;

    fn destructure(&mut self) -> (&mut FnMut(&mut Control), &mut Self::Args) {
        let MonomeTask {
            callback,
            args,
        } = self;

        (callback, args)
    }
}

type BelaApp<'a> = Bela<AppData<'a, MLRRenderer>>;

fn go() -> Result<(), bela::error::Error> {
    println!("loading samples & decoding...");

    let (mlr, renderer) = MLR::new(128., 44100);
    let monome = Monome::new("/prefix".to_string()).unwrap();

    println!("ok!");

    let mut monome_task = MonomeTask {
        callback: |control: &mut Control| {
            let mut grid = [0 as u8; 128];
            let monome = &mut control.monome;
            let mlr = &mut control.mlr;
            loop {
                match monome.poll() {
                    Some(e) => {
                        mlr.input(e);
                    }
                    _ => { }
                }
                mlr.main_thread_work();
                mlr.render_framebuffer(&mut grid);
                monome.set_all_intensity(&grid.to_vec());

                grid.iter_mut().map(|x| *x = 0).count();

                thread::sleep(Duration::from_millis(16));
            }
        },
        args: Control { mlr, monome }
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
        let mut output = context.audio_out();
        renderer.render_audio(&mut output);
        renderer.update_clock(frames * 2);
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
