use std::fs;
use std::fs::DirEntry;
use std::env;
use std::cmp;
use std::thread;
use std::time::Duration;
use std::ops::Index;

extern crate audrey;
extern crate pretty_env_logger;
#[macro_use]
extern crate log;
extern crate cubeb;

struct MLRSample
{
  name: String,
  channels: u32,
  rate: u32,
  data: Vec<f32>
}

impl MLRSample
{
    fn new(path: &DirEntry) -> MLRSample {
        info!("Loading {:?}...", path.path());
        let mut file = audrey::read::open(&path.path()).unwrap();
        let desc = file.description();
        let data: Vec<f32> = file.samples().map(Result::unwrap).collect::<Vec<_>>();
        let s = MLRSample {
            name: path.path().to_str().unwrap().to_string(),
            channels: desc.channel_count(),
            rate: desc.sample_rate(),
            data
        };

        info!("Loaded file: {} channels: {}, duration: {}, rate: {}", s.name(), s.channels(), s.duration(), s.rate());

        return s;
    }
    fn channels(&self) -> u32 {
        self.channels
    }
    fn frames(&self) -> usize {
        self.data.len() / self.channels as usize
    }
    fn duration(&self) -> f32 {
        (self.data.len() as f32) / self.channels as f32 / self.rate as f32
    }
    fn rate(&self) -> u32 {
        self.rate
    }
    fn name(&self) -> &str {
        &self.name
    }
}

impl Index<usize> for MLRSample {
    type Output = f32;

    fn index(&self, index: usize) -> &f32 {
        &self.data[index]
    }
}

enum PlaybackDirection {
    FORWARD,
    BACKWARD
}

struct MLRTrack
{
    row_index: u32,
    playing: bool,
    current_pos: usize,
    sample: MLRSample,
    start_index: u32, // [0, 15]
    stop_index: u32, // [0, 15], strictly greater than start
    direction: PlaybackDirection
}

impl MLRTrack
{
    fn new(row_index: u32, sample: MLRSample) -> MLRTrack {
        MLRTrack {
            row_index,
            playing: false,
            current_pos: 0,
            sample,
            start_index: 0,
            stop_index: 16,
            direction: PlaybackDirection::FORWARD
        }
    }
    fn start(&mut self) {
        self.playing = true;
    }
    fn stop(&mut self) {
        self.playing = false;
    }
    fn set_start(&mut self, index: u32) {
        self.start_index = index;
    }
    fn set_stop(&mut self, index: u32) {
        self.stop_index = index;
    }
    fn set_direction(&mut self,direction: PlaybackDirection) {
        self.direction = direction;
    }
    fn index_to_frames(sample: &MLRSample, index: u32) -> usize {
        (index * (sample.frames() as u32 / 16)) as usize
    }
    fn silence(data: &mut Vec<f32>) {
        data.iter_mut().for_each(move |v| *v = 0.0 );
    }
    // the right size for data is passed, in samples
    fn extract(&mut self, data: &mut Vec<f32>) {
      if !self.playing {
          Self::silence(data);
          return;
      }
      let mut remaining_frames = data.len() as u32 / self.sample.channels;
      let end_in_frames = Self::index_to_frames(&self.sample, self.stop_index) - 1;
      let mut buf_offset = 0;
      while remaining_frames != 0 {
        data[buf_offset] = self.sample[self.current_pos];
        self.current_pos += 1;
        if self.current_pos > end_in_frames {
            self.current_pos = Self::index_to_frames(&self.sample, self.start_index);
        }
        remaining_frames -= 1;
        buf_offset+=1;
      }
    }
}

struct Mixer<'a> {
  out: &'a mut [f32],
}

impl<'a> Mixer<'a> {
  fn new(out: &'a mut [f32]) -> Mixer<'a> {
      for i in 0..out.len() {
          out[i] = 0.0;
      }
      Mixer {
          out
      }
  }
  fn mix(&mut self, input: &Vec<f32>) {
      for i in 0..input.len() {
          self.out[i] += input[i];
      }
  }
}

fn usage() {
  println!("USAGE: mlr-rs <directory containing samples>");
}

fn validate_files(files: &Vec<MLRSample>) -> bool {
    let first = &files[0];
    let rate = first.rate();
    let duration = first.frames();

    for f in files.iter().skip(1) {
        if rate != f.rate() {
            error!("rate issue with {}: expected {} but found {}", f.name(), rate, f.rate());
            return false;
        }
        let longest = cmp::max(duration, f.frames());
        let shortest = cmp::min(duration, f.frames());
        if longest % shortest != 0 {
            error!("duration issue with {}: expected a multiple or divisor of {} but found {}", f.name(), duration, f.frames());
            return false;
        }
    }
    return true;
}

fn main() {
    pretty_env_logger::init();

    // read all files from $1 directory and load samples
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        usage();
    }

    let paths = fs::read_dir(args[1].clone()).unwrap();

    let mut samples: Vec<MLRSample> = Vec::new();

    for path in paths {
        let path = path.unwrap();
        samples.push(MLRSample::new(&path));
    }

    if !validate_files(&samples) {
        std::process::exit(1);
    }

    let common_rate = samples[0].rate();

    let mut tracks: Vec<MLRTrack> = Vec::new();
    let mut row = 0;
    for i in 0..samples.len() {
      let s = samples.remove(0);
      tracks.push(MLRTrack::new(row, s));
      row += 1;
    }

    // set up audio output
    let ctx = cubeb::init("mlr-rs").expect("Failed to create cubeb context");
    let params = cubeb::StreamParamsBuilder::new()
        .format(cubeb::SampleFormat::Float32NE)
        .rate(common_rate)
        .channels(1)
        .layout(cubeb::ChannelLayout::MONO)
        .take();

    let mut position = 0u32;

    let mut buf = vec![0.0 as f32; 256];
    tracks[0].start();
    tracks[1].start();
    // tracks[2].start();
    tracks[3].start();
    tracks[4].start();

    let mut builder = cubeb::StreamBuilder::new();
    builder
        .name("mlr-rs")
        .default_output(&params)
        .latency(256)
        .data_callback(move |input: &[f32], output| {
            {
                let mut m = Mixer::new(output);

                for i in tracks.iter_mut() {
                    i.extract(&mut buf);
                    m.mix(&buf);
                }
            }
            output.len() as isize
        })
    .state_callback(|state| {
        println!("stream {:?}", state);
    });

    let stream = builder.init(&ctx).expect("Failed to create cubeb stream");

    stream.start().unwrap();
    loop {
        thread::sleep(Duration::from_millis(5000));
    }

}
