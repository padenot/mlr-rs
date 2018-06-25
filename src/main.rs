extern crate audrey;
extern crate pretty_env_logger;
#[macro_use]
extern crate log;
extern crate cubeb;
extern crate monome;

use std::fs;
use std::fs::DirEntry;
use std::env;
use std::cmp;
use std::thread;
use std::sync::mpsc::channel;
use std::time::Duration;
use std::ops::Index;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};

use monome::{Monome, MonomeEvent, KeyDirection};

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

#[derive(Debug, PartialEq)]
enum PlaybackDirection {
    FORWARD,
    BACKWARD
}

struct MLRTrack
{
    row_index: usize,
    playing: bool,
    current_pos: isize,
    sample: MLRSample,
    start_index: u32, // [0, 15]
    stop_index: u32, // [0, 15], strictly greater than start
    direction: PlaybackDirection,
    gain: f32
}

struct MLRTrackMetadata
{
    row_index: usize,
    playing: bool,
    current_pos: isize,
    start_index: isize, // [0, 15]
    stop_index: isize, // [0, 15], strictly greater than start
    direction: PlaybackDirection,
    frames: isize,
    name: String
}

impl MLRTrackMetadata
{
    fn update_pos(&mut self, diff: isize) {
        if !self.playing {
            return;
        }
        let min = self.index_to_frames(self.start_index);
        let max = self.index_to_frames(self.stop_index);
        self.current_pos += if self.direction == PlaybackDirection::FORWARD { diff } else { -diff };
        if self.current_pos > max && self.direction == PlaybackDirection::FORWARD {
            self.current_pos = min + (self.current_pos - max);
        }
        if self.current_pos < min && self.direction == PlaybackDirection::BACKWARD {
            self.current_pos = max + (self.current_pos - min);
        }
    }
    fn enabled(&self) -> bool {
        self.playing
    }
    fn start(&mut self) {
        self.playing = true;
    }
    fn stop(&mut self) {
        self.playing = false;
    }
    fn set_start(&mut self, index: isize) {
        self.start_index = index;
    }
    fn set_stop(&mut self, index: isize) {
        self.stop_index = index;
    }
    fn set_head(&mut self, index: isize) {
        self.current_pos = self.index_to_frames(index);
    }
    fn set_direction(&mut self,direction: PlaybackDirection) {
        self.direction = direction;
    }
    fn frames_to_index(&self, frames: isize) -> isize {
        frames * 16 / self.frames
    }
    fn index_to_frames(&self, index: isize) -> isize{
        (self.frames / 16) * index
    }
    fn current_led(&self) -> isize {
        let c = self.frames_to_index(self.current_pos);
        c
    }
    fn row_index(&self) -> usize {
        self.row_index
    }
    fn name(&self) -> &str {
        &self.name
    }
}

impl MLRTrack
{
    fn new(row_index: usize, sample: MLRSample) -> MLRTrack {
        MLRTrack {
            row_index,
            playing: false,
            current_pos: 0,
            sample,
            start_index: 0,
            stop_index: 16,
            direction: PlaybackDirection::FORWARD,
            gain: 1.0
        }
    }
    fn metadata(&self) -> MLRTrackMetadata {
        MLRTrackMetadata {
            row_index: self.row_index,
            playing: false,
            current_pos: 0,
            start_index: 0,
            stop_index: 16,
            frames: self.sample.frames() as isize,
            direction: PlaybackDirection::FORWARD,
            name: self.name().clone().to_string()
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
    fn set_head(&mut self, index: u32) {
        self.current_pos = Self::index_to_frames(&self.sample, index);
    }
    fn set_direction(&mut self, direction: PlaybackDirection) {
        self.direction = direction;
    }
    fn adjust_gain(&mut self, gain_delta: i32) {
      self.gain += gain_delta as f32 / 10.;
    }
    fn index_to_frames(sample: &MLRSample, index: u32) -> isize {
        (index * (sample.frames() as u32 / 16)) as isize
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
      let begining_in_frames = Self::index_to_frames(&self.sample, self.start_index);
      let mut buf_offset = 0;
      while remaining_frames != 0 {
        data[buf_offset] = self.sample[self.current_pos as usize] * self.gain;
        self.current_pos += if self.direction == PlaybackDirection::FORWARD { 1 } else { -1 };
        // println!("{} in [{}, {}]", self.current_pos, begining_in_frames, end_in_frames);
        if self.current_pos > end_in_frames && self.direction == PlaybackDirection::FORWARD {
            self.current_pos = begining_in_frames;
        }
        if self.current_pos < begining_in_frames && self.direction == PlaybackDirection::BACKWARD {
            self.current_pos = end_in_frames;
        }
        remaining_frames -= 1;
        buf_offset+=1;
      }
    }
    fn name(&self) -> &str {
        self.sample.name()
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


struct ClockUpdater {
    clock: Arc<AtomicUsize>
}

impl ClockUpdater {
    fn increment(&mut self, frames: usize) {
        self.clock.store(self.clock.load(Ordering::Relaxed) + frames, Ordering::Relaxed);
    }
}

struct ClockConsumer {
    clock: Arc<AtomicUsize>,
    rate: u32,
    tempo: f32
}

impl ClockConsumer {
    fn raw_frames(&self) -> usize {
        self.clock.load(Ordering::Relaxed)
    }
    fn beat(&self) -> f32 {
        self.tempo / 60. * (self.raw_frames() as f32 / self.rate as f32)
    }
}

fn clock(tempo: f32, rate: u32) -> (ClockUpdater, ClockConsumer) {
    let c = Arc::new(AtomicUsize::new(0));
    (ClockUpdater { clock: c.clone() }, ClockConsumer { clock: c, tempo, rate })
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

#[derive(Clone, PartialEq)]
enum MLRIntent
{
    Nothing,
    Trigger,
    Loop
}

enum MLRAction
{
    Nothing,
    Trigger((usize, usize)),
    Loop((usize, usize, usize)),
    GainChange((usize, i32)),
    TrackStatus(usize),
    Pattern(usize)
}

struct GridStateTracker {
    buttons: Vec<MLRIntent>,
    width: usize,
    height: usize
}


// state tracker for grid action on the botton rows
impl GridStateTracker {
    fn new(width: usize, height: usize) -> GridStateTracker {
      GridStateTracker {
          width,
          height,
          buttons: vec![MLRIntent::Nothing; width * height]
      }
    }
    fn down(&mut self, x: usize, y: usize) {
      if y == 0 { // control row
          self.buttons[Self::idx(self.width, x, y)] = MLRIntent::Trigger;
      } else { // track rows
          // If, when pressing down, we find another button on the same line already down, this is a
          // loop.
          let mut foundanother = false;
          for i  in 0..self.width {
              if self.buttons[Self::idx(self.width, i, y)] != MLRIntent::Nothing {
                  self.buttons[Self::idx(self.width, i, y)] = MLRIntent::Loop;
                  self.buttons[Self::idx(self.width, x, y)] = MLRIntent::Loop;
                  foundanother = true;
              }
          }
          if !foundanother {
              self.buttons[Self::idx(self.width, x, y)] = MLRIntent::Trigger;
          }
      }
    }
    fn up(&mut self, x: usize, y: usize) -> MLRAction {
        if y == 0 { // control row
            // is mod 1 (8th button) or mod 2 (9th button) down
            let mut gain_delta = 0;
            let mut pattern = 0;
            if self.buttons[Self::idx(self.width, 7, y)] != MLRIntent::Nothing {
                gain_delta = -1;
            }
            if self.buttons[Self::idx(self.width, 8, y)] != MLRIntent::Nothing {
                gain_delta = 1;
            }
            if self.buttons[Self::idx(self.width, 9, y)] != MLRIntent::Nothing {
                pattern = 1;
            }
            if self.buttons[Self::idx(self.width, 10, y)] != MLRIntent::Nothing {
                pattern = 2;
            }
            self.buttons[Self::idx(self.width, x, y)] = MLRIntent::Nothing;
            if gain_delta != 0 {
                return MLRAction::GainChange((x, gain_delta))
            }
            if pattern != 0 {
                return MLRAction::Pattern(pattern)
            }

            MLRAction::TrackStatus(x)
        } else { // tracks rows
            match self.buttons[Self::idx(self.width, x, y)].clone() {
                MLRIntent::Nothing => { panic!("what."); }
                MLRIntent::Trigger => {
                    self.buttons[Self::idx(self.width, x, y)] = MLRIntent::Nothing;
                    MLRAction::Trigger((x ,y))
                }
                MLRIntent::Loop => {
                    // Find the other button that is down, if we find one that if intended for looping,
                    // loop between the two points. Otherwise, it's just the second loop point that is
                    // being released.
                    let mut other: Option<usize> = None;
                    for i in 0..self.width {
                        if i != x && self.buttons[Self::idx(self.width, i, y)] == MLRIntent::Loop {
                            other = Some(i);
                        }
                    }

                    self.buttons[Self::idx(self.width, x, y)] = MLRIntent::Nothing;

                    match other {
                        Some(i) => MLRAction::Loop((y, x, i)),
                        None => MLRAction::Nothing
                    }
                }
            }
        }
    }
    fn idx(width: usize, x: usize, y: usize) -> usize {
      y * width + x
    }
    fn dump(&self) {
        for i in 0..self.height {
            for j in 0..self.width {
                match self.buttons[Self::idx(self.width, j, i)] {
                    MLRIntent::Nothing => { print!(" "); }
                    MLRIntent::Trigger => { print!("T"); }
                    MLRIntent::Loop => { print!("L"); }
                }
            }
            println!(" ");
        }
    }
}

#[derive(Debug)]
enum Message {
    Enable((usize)),
    Disable((usize)),
    SetHead((usize, u32)),
    SetStart((usize, u32)),
    SetEnd((usize, u32)),
    SetDirection((usize, PlaybackDirection)),
    GainChange((usize, i32))
}

fn main() {
    pretty_env_logger::init();

    // read all files from $1 directory and load samples
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        usage();
        return;
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
    let mut tracks_meta: Vec<MLRTrackMetadata> = Vec::new();
    let mut row = 0;
    for _i in 0..samples.len() {
      let s = samples.remove(0);
      let t = MLRTrack::new(row, s);
      tracks_meta.push(t.metadata());
      tracks.push(t);
      row += 1;
    }

    let (sender, receiver) = channel::<Message>();
    let (mut clock_updater, clock_receiver) = clock(128., 48000);
    let rx = Arc::new(Mutex::new(receiver));

    // set up audio output
    let ctx = cubeb::init("mlr-rs").expect("Failed to create cubeb context");
    let params = cubeb::StreamParamsBuilder::new()
        .format(cubeb::SampleFormat::Float32NE)
        .rate(common_rate)
        .channels(1)
        .layout(cubeb::ChannelLayout::MONO)
        .take();

    let mut buf = vec![0.0 as f32; 256];

    let mut builder = cubeb::StreamBuilder::new();
    builder
        .name("mlr-rs")
        .default_output(&params)
        .latency(256)
        .data_callback(move |_input: &[f32], output| {
            loop {
                match rx.lock().unwrap().try_recv() {
                    Ok(msg) => {
                        match msg {
                          Message::Enable(track) => {
                              if track as usize > tracks.len() {
                                  info!("msg to track not loaded");
                                  continue;
                              }
                              tracks[track as usize].start();
                              info!("starting {}", tracks[track as usize].name());
                          }
                          Message::Disable(track) => {
                              if track as usize > tracks.len() {
                                  info!("msg to track not loaded");
                                  continue;
                              }
                              tracks[track as usize].stop();
                              info!("stopping {}", tracks[track as usize].name());
                          }
                          Message::SetHead((track, pos)) => {
                              if track as usize > tracks.len() {
                                  info!("msg to track not loaded");
                                  continue;
                              }
                              tracks[track as usize].set_head(pos);
                              info!("setting head for {} at {}", tracks[track as usize].name(), pos);
                          }
                          Message::SetStart((track, pos)) => {
                              if track as usize > tracks.len() {
                                  info!("msg to track not loaded");
                                  continue;
                              }
                              info!("setting start point for {} at {}", tracks[track as usize].name(), pos);
                              tracks[track as usize].set_start(pos);
                          }
                          Message::SetEnd((track, pos)) => {
                              if track as usize > tracks.len() {
                                  info!("msg to track not loaded");
                                  continue;
                              }
                              info!("setting end point for {} at {}", tracks[track as usize].name(), pos);
                              tracks[track as usize].set_stop(pos);
                          }
                          Message::SetDirection((track, direction)) => {
                              if track as usize > tracks.len() {
                                  info!("msg to track not loaded");
                                  continue;
                              }
                              info!("setting direction for {} at to {:?}", tracks[track as usize].name(), direction);
                              tracks[track as usize].set_direction(direction);
                          }
                          Message::GainChange((track, gain_delta)) => {
                              if track as usize > tracks.len() {
                                  info!("msg to track not loaded");
                                  continue;
                              }
                              info!("adjust gain for {}, delta {}", tracks[track as usize].name(), gain_delta);
                              tracks[track as usize].adjust_gain(gain_delta);
                          }
                          _ => {
                              error!("unexpected message.");
                          }
                      }
                  },
                  Err(err) => {
                      match err {
                          std::sync::mpsc::TryRecvError::Empty => {
                              break;
                          }
                          std::sync::mpsc::TryRecvError::Disconnected => {
                              error!("disconnected");
                          }
                      }
                  }
                }
            }
            {
                let mut m = Mixer::new(output);

                for i in tracks.iter_mut() {
                    i.extract(&mut buf);
                    m.mix(&buf);
                }
            }
            clock_updater.increment(output.len());
            output.len() as isize
        })
    .state_callback(|state| {
        info!("stream {:?}", state);
    });

    let stream = builder.init(&ctx).expect("Failed to create cubeb stream");

    let mut monome = Monome::new("/mlr-rs").unwrap();

    stream.start().unwrap();

    let mut tracks_enabled : Vec<bool> = vec![false; 7];
    let mut tracks_pos: Vec<usize> = vec![0; 7];

    monome.all(false);

    let mut prev_clock = 0;

    let mut leds: Vec<i32> = vec![-1; 7];
    let mut state_tracker = GridStateTracker::new(monome.width(), monome.height());

    loop {
        loop {
            match monome.poll() {
                Some(MonomeEvent::GridKey{x, y, direction}) => {
                    match direction {
                        KeyDirection::Down => {
                            state_tracker.down(x as usize, y as usize);
                        },
                        KeyDirection::Up => {
                            match state_tracker.up(x as usize, y as usize) {
                                MLRAction::Trigger((x, y)) => {
                                    let track_index = y - 1;
                                    println!("{} > {} ", track_index, tracks_meta.len());
                                    if track_index > tracks_meta.len() - 1 {
                                        break;
                                    }
                                    tracks_meta[track_index as usize].start();
                                    tracks_meta[track_index as usize].set_head(x as isize);
                                    monome.set((y - 1) as i32, 0, true);
                                    sender.send(Message::Enable((y - 1))).unwrap();
                                    sender.send(Message::SetHead((y - 1, x as u32))).unwrap();
                                    sender.send(Message::SetStart(((y - 1) as usize, 0))).unwrap();
                                    sender.send(Message::SetEnd(((y - 1) as usize, 16))).unwrap();
                                    sender.send(Message::SetDirection(((y - 1) as usize, PlaybackDirection::FORWARD))).unwrap();
                                    tracks_meta[track_index as usize].set_start(0 as isize);
                                    tracks_meta[track_index as usize].set_stop(16 as isize);
                                    tracks_meta[track_index as usize].set_direction(PlaybackDirection::FORWARD);
                                }
                                MLRAction::Loop((row, mut start, mut end)) => {
                                    let track_index = row - 1;
                                    if track_index > tracks_meta.len() - 1 {
                                        break;
                                    }
                                    if start > end {
                                        sender.send(Message::SetDirection(((y - 1) as usize, PlaybackDirection::BACKWARD))).unwrap();
                                        sender.send(Message::SetHead(((y - 1) as usize, end as u32))).unwrap();
                                        tracks_meta[track_index as usize].set_direction(PlaybackDirection::BACKWARD);
                                        tracks_meta[track_index as usize].set_head(end as isize);
                                    } else {
                                        sender.send(Message::SetDirection(((y - 1) as usize, PlaybackDirection::FORWARD))).unwrap();
                                        sender.send(Message::SetHead(((y - 1) as usize, start as u32))).unwrap();
                                        tracks_meta[track_index as usize].set_head(start as isize);
                                        tracks_meta[track_index as usize].set_direction(PlaybackDirection::FORWARD);
                                    }
                                    if start > end {
                                        std::mem::swap(&mut start, &mut end);
                                    }
                                    sender.send(Message::SetStart(((y - 1) as usize, start as u32))).unwrap();
                                    sender.send(Message::SetEnd(((y - 1) as usize, end as u32))).unwrap();
                                    tracks_meta[track_index as usize].set_start(start as isize);
                                    tracks_meta[track_index as usize].set_stop(end as isize);
                                    if !tracks_meta[track_index as usize].enabled() {
                                        tracks_meta[track_index as usize].start();
                                        sender.send(Message::Enable(track_index as usize)).unwrap();
                                    }
                                }
                                MLRAction::GainChange((track, gain_delta)) => {
                                    sender.send(Message::GainChange((track, gain_delta)));
                                }
                                MLRAction::TrackStatus(track) => {
                                    if track as usize > tracks_meta.len() {
                                        continue;
                                    }
                                    if tracks_meta[x as usize].enabled() {
                                        tracks_meta[x as usize].stop();
                                        monome.set(x as i32, 0, false);
                                        sender.send(Message::Disable(x as usize)).unwrap();
                                    } else {
                                        tracks_meta[x as usize].start();
                                        monome.set(x as i32, 0, true);
                                        sender.send(Message::Enable(x as usize)).unwrap();
                                    }
                                }
                                MLRAction::Pattern(pattern) => {
                                    info!("not implemented");
                                }
                                MLRAction::Nothing => {
                                }
                            }
                        }
                    }
                }
                Some(_) => {
                    break;
                }
                None => {
                    break;
                }
            }
        }
        let current_time = clock_receiver.raw_frames();
        let diff = (current_time - prev_clock)  as isize;
        for t in tracks_meta.iter_mut() {
            t.update_pos(diff);
            if t.enabled() {
                let current_led = t.current_led();
                if leds[t.row_index()] != current_led as i32 && t.enabled() {
                    monome.set(leds[t.row_index()] as i32, t.row_index() as i32 + 1, false);
                    leds[t.row_index()] = current_led as i32;
                    monome.set(t.current_led() as i32, t.row_index() as i32 + 1, true);
                }
            }
        }
        prev_clock = current_time;
        thread::sleep(Duration::from_millis(10));
    }

}
