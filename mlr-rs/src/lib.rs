extern crate audrey;
#[macro_use]
extern crate log;
extern crate audio_clock;
extern crate chrono;
extern crate monome;
extern crate timer;

use audrey::*;
use audio_clock::*;
use std::process;
use std::mem;
use std::sync;
use std::cmp;
use std::env;
use std::fs;
use std::fs::DirEntry;
use std::ops::Index;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use monome::{KeyDirection, Monome, MonomeEvent};

struct MLRSample {
    name: String,
    channels: u32,
    rate: u32,
    data: Vec<f32>,
}

impl MLRSample {
    fn new(path: &DirEntry) -> MLRSample {
        info!("Loading {:?}...", path.path());
        let mut file = audrey::read::open(&path.path()).unwrap();
        let desc = file.description();
        let data: Vec<f32> = file.samples().map(Result::unwrap).collect::<Vec<_>>();
        let s = MLRSample {
            name: path.path().to_str().unwrap().to_string(),
            channels: desc.channel_count(),
            rate: desc.sample_rate(),
            data,
        };

        info!(
            "Loaded file: {} channels: {}, duration: {}, rate: {}",
            s.name(),
            s.channels(),
            s.duration(),
            s.rate()
        );

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

#[derive(Debug, PartialEq, Clone)]
enum PlaybackDirection {
    FORWARD,
    BACKWARD,
}

struct MLRTrack {
    row_index: usize,
    playing: bool,
    current_pos: isize,
    sample: MLRSample,
    start_index: u32, // [0, 15]
    stop_index: u32,  // [0, 15], strictly greater than start
    direction: PlaybackDirection,
    gain: f32,
}

struct MLRTrackMetadata {
    row_index: usize,
    playing: bool,
    current_pos: isize,
    start_index: isize, // [0, 15]
    stop_index: isize,  // [0, 15], strictly greater than start
    direction: PlaybackDirection,
    frames: isize,
    name: String,
    ever_played: bool
}

impl MLRTrackMetadata {
    fn update_pos(&mut self, diff: isize) {
        if !self.playing {
            return;
        }
        let min = self.index_to_frames(self.start_index);
        let max = self.index_to_frames(self.stop_index);
        self.current_pos += if self.direction == PlaybackDirection::FORWARD {
            diff
        } else {
            -diff
        };
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
    fn ever_played(&self) -> bool {
        self.ever_played
    }
    fn start(&mut self) {
        self.ever_played = true;
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
        self.ever_played = true;
        self.current_pos = self.index_to_frames(index);
    }
    fn set_direction(&mut self, direction: PlaybackDirection) {
        self.direction = direction;
    }
    fn frames_to_index(&self, frames: isize) -> isize {
        frames * 16 / self.frames
    }
    fn index_to_frames(&self, index: isize) -> isize {
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
    fn begin(&self) -> isize {
        self.start_index
    }
    fn end(&self) -> isize {
        self.stop_index
    }
}

impl MLRTrack {
    fn new(row_index: usize, sample: MLRSample) -> MLRTrack {
        MLRTrack {
            row_index,
            playing: false,
            current_pos: 0,
            sample,
            start_index: 0,
            stop_index: 16,
            direction: PlaybackDirection::FORWARD,
            gain: 1.0,
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
            name: self.name().clone().to_string(),
            ever_played: false
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
        data.iter_mut().for_each(move |v| *v = 0.0);
    }
    fn extract(&mut self) -> f32 {
        if !self.playing {
            return 0.0;
        }
        let end_in_frames = Self::index_to_frames(&self.sample, self.stop_index) - 1;
        let begining_in_frames = Self::index_to_frames(&self.sample, self.start_index);
        let data = self.sample[self.current_pos as usize] * self.gain;
        self.current_pos += if self.direction == PlaybackDirection::FORWARD {
            1
        } else {
            -1
        };
        if self.current_pos > end_in_frames && self.direction == PlaybackDirection::FORWARD {
            self.current_pos = begining_in_frames;
        }
        if self.current_pos < begining_in_frames && self.direction == PlaybackDirection::BACKWARD {
            self.current_pos = end_in_frames;
        }
        return data;
    }
    fn name(&self) -> &str {
        self.sample.name()
    }
}

struct Mixer<'a> {
    out: &'a mut [f32],
    offset: usize,
}

impl<'a> Mixer<'a> {
    fn new(out: &'a mut [f32]) -> Mixer<'a> {
        for i in 0..out.len() {
            out[i] = 0.0;
        }
        Mixer { out, offset: 0 }
    }
    fn mix(&mut self, input: &Vec<f32>) {
        assert!(self.offset == 0);
        for i in 0..input.len() {
            self.out[i] += input[i];
        }
    }
    fn mix_sample(&mut self, input: f32) {
        self.out[self.offset] += input;
    }
    fn next(&mut self) -> bool {
        if self.offset == self.out.len() - 1 {
            return false;
        }
        self.offset += 1;
        return true;
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
            error!(
                "rate issue with {}: expected {} but found {}",
                f.name(),
                rate,
                f.rate()
            );
            return false;
        }
        let longest = cmp::max(duration, f.frames());
        let shortest = cmp::min(duration, f.frames());
        if longest % shortest != 0 {
            error!(
                "duration issue with {}: expected a multiple or divisor of {} but found {}",
                f.name(),
                duration,
                f.frames()
            );
            return false;
        }
    }
    return true;
}

#[derive(Clone, PartialEq)]
enum MLRIntent {
    Nothing,
    Trigger,
    Loop,
}
#[derive(Debug, Copy, Clone)]
enum MLRAction {
    Nothing,
    Trigger((usize, usize)),
    Loop((usize, usize, usize)),
    GainChange((usize, i32)),
    TrackStatus(usize),
    Pattern(u8),
}

struct GridStateTracker {
    buttons: Vec<MLRIntent>,
    width: usize,
    height: usize,
}

// state tracker for grid action on the botton rows
impl GridStateTracker {
    fn new(width: usize, height: usize) -> GridStateTracker {
        GridStateTracker {
            width,
            height,
            buttons: vec![MLRIntent::Nothing; width * height],
        }
    }
    fn down(&mut self, x: usize, y: usize) {
        if y == 0 {
            // control row
            self.buttons[Self::idx(self.width, x, y)] = MLRIntent::Trigger;
        } else {
            // track rows
            // If, when pressing down, we find another button on the same line already down, this is a
            // loop.
            let mut foundanother = false;
            for i in 0..self.width {
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
        if y == 0 {
            // 0 to 7 are system buttons.
            // 8 to 14 are enable/disable of tracks
            // 15 triggers pattern
            if self.buttons[Self::idx(self.width, 15, y)] != MLRIntent::Nothing {
                self.buttons[Self::idx(self.width, x, y)] = MLRIntent::Nothing;
                return MLRAction::Pattern(0);
            }
            self.buttons[Self::idx(self.width, x, y)] = MLRIntent::Nothing;
            // offset by 8 to find the track index
            MLRAction::TrackStatus(x - 8)
        } else {
            // tracks rows
            match self.buttons[Self::idx(self.width, x, y)].clone() {
                MLRIntent::Nothing => {
                    /* someone pressed a key during startup */
                    MLRAction::Nothing
                }
                MLRIntent::Trigger => {
                    self.buttons[Self::idx(self.width, x, y)] = MLRIntent::Nothing;
                    MLRAction::Trigger((x, y))
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
                        None => MLRAction::Nothing,
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
                    MLRIntent::Nothing => {
                        print!(" ");
                    }
                    MLRIntent::Trigger => {
                        print!("T");
                    }
                    MLRIntent::Loop => {
                        print!("L");
                    }
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
    GainChange((usize, i32)),
}

pub struct MLR {
    sender: Sender<Message>,
    tracks_meta: Vec<MLRTrackMetadata>,
    previous_time: usize,
    leds: Vec<i32>,
    monome: Monome,
    audio_clock: ClockConsumer,
    pattern: Vec<(usize, MLRAction)>,
    recording_end: Option<usize>,
    pattern_duration_frames: usize,
    recording_pattern: bool,
    pattern_playback: bool,
    pattern_index: usize,
    pattern_rec_time_end: usize,
    state_tracker: GridStateTracker,
    rate: u32,
}

impl MLR {
    pub fn new(tempo: f32, rate: u32) -> (MLR, MLRRenderer) {
        let (clock_updater, clock_consumer) = audio_clock(tempo, rate);
        let (sender, receiver) = channel::<Message>();
        let rx = Arc::new(Mutex::new(receiver));

        let paths = fs::read_dir("mlr-samples").unwrap();

        let mut samples: Vec<MLRSample> = Vec::new();

        for path in paths {
            let path = path.unwrap();
            samples.push(MLRSample::new(&path));
        }

        if !validate_files(&samples) {
            process::exit(1);
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

        let mut channel = 0;

        let mut renderer = MLRRenderer::new(tracks, rx, clock_updater, channel);

        let state_tracker = GridStateTracker::new(16, 8);

        let leds = vec![-1; 7];

        let mut monome = Monome::new("/mlr-rs").unwrap();
        monome.all(false);

        (
            MLR {
                rate,
                sender,
                tracks_meta,
                previous_time: 0,
                leds,
                monome,
                audio_clock: clock_consumer,
                pattern: Vec::new(),
                recording_end: None,
                pattern_duration_frames: (60. / 128. * 4. * 48000.) as usize,
                recording_pattern: false,
                pattern_playback: false,
                pattern_index: 0,
                pattern_rec_time_end: 0,
                state_tracker,
            },
            renderer,
        )
    }
    fn track_length(&self) -> usize {
        self.monome.width()
    }
    fn track_max(&self) -> usize {
        self.monome.height()
    }
    fn start(&mut self, track_index: usize) {
        if !self.tracks_meta[track_index].enabled() {
            self.sender.send(Message::Enable(track_index)).unwrap();
            self.tracks_meta[track_index as usize].start();
            self.monome.set(track_index as i32, 0, true);
        }
    }
    fn stop(&mut self, track_index: usize) {
        if self.tracks_meta[track_index].enabled() {
            self.sender.send(Message::Disable(track_index)).unwrap();
            self.tracks_meta[track_index as usize].stop();
            self.monome.set(track_index as i32, 0, false);
        }
    }
    fn set_head(&mut self, track_index: usize, head_pos: isize) {
        self.sender
            .send(Message::SetHead((track_index, head_pos as u32)))
            .unwrap();
        self.tracks_meta[track_index as usize].set_head(head_pos as isize);
        self.monome.set(track_index as i32, 0, true);
    }
    fn set_start(&mut self, track_index: usize, start: usize) {
        self.sender
            .send(Message::SetStart((track_index, start as u32)))
            .unwrap();
        self.tracks_meta[track_index].set_start(start as isize);
    }
    fn set_end(&mut self, track_index: usize, end: usize) {
        self.sender
            .send(Message::SetEnd((track_index, end as u32)))
            .unwrap();
        self.tracks_meta[track_index].set_stop(end as isize);
    }
    fn set_direction(&mut self, track_index: usize, direction: PlaybackDirection) {
        self.sender
            .send(Message::SetDirection((track_index, direction.clone())))
            .unwrap();
        self.tracks_meta[track_index].set_direction(direction);
    }
    fn change_gain(&mut self, track_index: usize, gain_delta: i32) {
        self.sender
            .send(Message::GainChange((track_index, gain_delta)))
            .unwrap();
    }
    fn enabled(&mut self, track_index: usize) -> bool {
        self.tracks_meta[track_index].enabled()
    }
    fn track_loaded(&mut self, track_index: usize) -> bool {
        track_index < self.tracks_meta.len()
    }
    fn update_leds(&mut self, grid: &mut [u8; 128]) {
        if self.recording_pattern {
            grid[15] = 15;
        }
        if self.pattern_playback {
            grid[15] = 8;
        }
        let track_length = self.track_length();
        for i in 0..self.tracks_meta.len() {
            let t = &mut self.tracks_meta[i];
            let idx: usize = (i+1) * track_length + t.current_led() as usize;
            grid[idx] = if t.ever_played() { 15 } else { 0 };
            grid[8 + i] = if t.enabled() {
                15
            } else {
                if t.ever_played() {
                    5
                } else {
                    0
                }
            };
        }
    }
    fn reset_pattern(&mut self) {
        self.pattern.clear();
        self.recording_end = None;
        self.pattern_duration_frames = (60. / 128. * 4. * self.rate as f32) as usize;
        self.recording_pattern = false;
        self.pattern_playback = false;
        self.pattern_index = 0;
        self.pattern_rec_time_end = 0;
    }
    fn handle_action(&mut self, action: MLRAction) {
        // If recording pattern but button has been pressed, stop recording a pattern.
        match action {
            MLRAction::Pattern(pattern) => {
                self.reset_pattern();
            }
            _ => {}
        }

        if self.recording_pattern {
            if self.pattern.len() == 0 {
                let begin = self.audio_clock.raw_frames();
                self.recording_end = Some(begin + self.pattern_duration_frames);
                self.pattern.clear();
            }
            self.pattern.push((self.audio_clock.raw_frames(), action));
        }

        match action {
            MLRAction::Trigger((x, y)) => {
                let track_index = y - 1;
                if !self.track_loaded(track_index) {
                    return;
                }
                self.start(track_index);
                self.set_head(track_index, x as isize);
                self.set_start(track_index, 0);
                self.set_end(track_index, 16);
                self.set_direction(track_index, PlaybackDirection::FORWARD);
            }
            MLRAction::Loop((row, mut start, mut end)) => {
                let track_index = row - 1;
                if !self.track_loaded(track_index) {
                    return;
                }
                if start > end {
                    self.set_direction(track_index, PlaybackDirection::BACKWARD);
                    self.set_head(track_index, end as isize);
                } else {
                    self.set_direction(track_index, PlaybackDirection::FORWARD);
                    self.set_head(track_index, start as isize);
                }
                if start > end {
                    mem::swap(&mut start, &mut end);
                }
                self.set_start(track_index, start);
                self.set_end(track_index, end);
                self.start(track_index);
            }
            MLRAction::GainChange((track_index, gain_delta)) => {
                if !self.track_loaded(track_index) {
                    return;
                }
                self.change_gain(track_index, gain_delta);
            }
            MLRAction::TrackStatus(track_index) => {
                if !self.track_loaded(track_index) {
                    return;
                }
                if self.enabled(track_index as usize) {
                    self.stop(track_index as usize);
                } else {
                    self.start(track_index as usize);
                }
            }
            MLRAction::Pattern(pattern) => {
                self.reset_pattern();
                self.recording_pattern = true;
            }
            MLRAction::Nothing => {}
        }
    }

    pub fn main_thread_work(&mut self) {
        let current_time = self.audio_clock.raw_frames();
        let diff = (current_time - self.previous_time) as isize;
        for i in 0..self.tracks_meta.len() {
            let t = &mut self.tracks_meta[i];
            t.update_pos(diff);
        }
        self.previous_time = current_time;

        if self.pattern_playback {
            debug!(
                "pattern[{}].begin: {:?}, offset: {}, end: {}",
                self.pattern_index,
                self.pattern[self.pattern_index].0,
                self.audio_clock.raw_frames() - self.pattern_rec_time_end,
                self.pattern_duration_frames
            );
            // this is the clock time, between 0 and the duration of a row, 0 being the
            // start time of a loop of a pattern.
            let clock_in_pattern = (self.audio_clock.raw_frames() - self.pattern_rec_time_end)
                % self.pattern_duration_frames;
            // if the current time is later than the current action time in the pattern,
            // and we're not playing after the last pattern item (waiting to loop, pattern
            // index being zero and clock time being past the last item start time),
            // trigger the action and move to the next pattern item. If we're past the end,
            // wrap around in the pattern array.
            if clock_in_pattern > self.pattern[self.pattern_index].0
                && !(self.pattern_index == 0 && clock_in_pattern > self.pattern.last().unwrap().0)
            {
                let action = self.pattern[self.pattern_index].1.clone();
                self.pattern_index += 1;
                if self.pattern_index == self.pattern.len() {
                    self.pattern_index = 0;
                }
                self.handle_action(action);
            }
        }

        match self.recording_end {
            Some(end) => {
                if end < self.audio_clock.raw_frames() {
                    self.pattern_rec_time_end = self.audio_clock.raw_frames();
                    self.recording_pattern = false;
                    self.recording_end = None;
                    self.pattern_playback = true;
                    let offset = self.pattern[0].0;
                    // normalize pattern with a 0 start
                    for i in 0..self.pattern.len() {
                        self.pattern[i].0 -= offset;
                    }
                }
            }
            _ => {}
        }
    }
    pub fn handle_inputs(&mut self) {
        loop {
            match self.monome.poll() {
                Some(e) => { self.input(e) }
                _ => { return; }
            }
        }
    }
    pub fn input(&mut self, event: MonomeEvent) {
        match event {
            MonomeEvent::GridKey { x, y, direction } => match direction {
                KeyDirection::Down => {
                    self.state_tracker.down(x as usize, y as usize);
                }
                KeyDirection::Up => {
                    let action = self.state_tracker.up(x as usize, y as usize);
                    self.handle_action(action);
                }
            }
            _ => { return; }
        }
    }
    pub fn render_framebuffer(&mut self, grid: &mut [u8; 128]) {
        self.update_leds(grid);
    }
    pub fn render(&mut self) {
        let mut leds = [0 as u8; 128];
        self.update_leds(&mut leds);
        self.monome.set_all_intensity(&leds);
    }
}

pub struct MLRRenderer {
    tracks: Vec<MLRTrack>,
    rx: sync::Arc<Mutex<sync::mpsc::Receiver<Message>>>,
    clock_updater: ClockUpdater,
    channel: usize
}

impl MLRRenderer {
    fn new(
        tracks: Vec<MLRTrack>,
        rx: sync::Arc<Mutex<sync::mpsc::Receiver<Message>>>,
        clock_updater: ClockUpdater,
        channel: usize
    ) -> MLRRenderer {
        MLRRenderer { tracks, rx, clock_updater, channel }
    }
    pub fn update_clock(&mut self, frames: usize) {
        self.clock_updater.increment(frames);
    }
    pub fn render_audio(&mut self, output: &mut [f32]) {
        let mut m = Mixer::new(output);
        loop {
            match self.rx.lock().unwrap().try_recv() {
                Ok(msg) => match msg {
                    Message::Enable(track) => {
                        if track as usize > self.tracks.len() {
                            info!("msg to track not loaded");
                            continue;
                        }
                        self.tracks[track as usize].start();
                        info!("starting {}", self.tracks[track as usize].name());
                    }
                    Message::Disable(track) => {
                        if track as usize > self.tracks.len() {
                            info!("msg to track not loaded");
                            continue;
                        }
                        self.tracks[track as usize].stop();
                        info!("stopping {}", self.tracks[track as usize].name());
                    }
                    Message::SetHead((track, pos)) => {
                        if track as usize > self.tracks.len() {
                            info!("msg to track not loaded");
                            continue;
                        }
                        self.tracks[track as usize].set_head(pos);
                        info!(
                            "setting head for {} at {}",
                            self.tracks[track as usize].name(),
                            pos
                        );
                    }
                    Message::SetStart((track, pos)) => {
                        if track as usize > self.tracks.len() {
                            info!("msg to track not loaded");
                            continue;
                        }
                        info!(
                            "setting start point for {} at {}",
                            self.tracks[track as usize].name(),
                            pos
                        );
                        self.tracks[track as usize].set_start(pos);
                    }
                    Message::SetEnd((track, pos)) => {
                        if track as usize > self.tracks.len() {
                            info!("msg to track not loaded");
                            continue;
                        }
                        info!(
                            "setting end point for {} at {}",
                            self.tracks[track as usize].name(),
                            pos
                        );
                        self.tracks[track as usize].set_stop(pos);
                    }
                    Message::SetDirection((track, direction)) => {
                        if track as usize > self.tracks.len() {
                            info!("msg to track not loaded");
                            continue;
                        }
                        info!(
                            "setting direction for {} at to {:?}",
                            self.tracks[track as usize].name(),
                            direction
                        );
                        self.tracks[track as usize].set_direction(direction);
                    }
                    Message::GainChange((track, gain_delta)) => {
                        if track as usize > self.tracks.len() {
                            info!("msg to track not loaded");
                            continue;
                        }
                        info!(
                            "adjust gain for {}, delta {}",
                            self.tracks[track as usize].name(),
                            gain_delta
                        );
                        self.tracks[track as usize].adjust_gain(gain_delta);
                    }
                },
                Err(err) => match err {
                    sync::mpsc::TryRecvError::Empty => {}
                    sync::mpsc::TryRecvError::Disconnected => {
                        error!("disconnected");
                    }
                },
            }
            let mut out: f32 = 0.0;
            for i in self.tracks.iter_mut() {
                m.mix_sample(i.extract());
            }
            if !m.next() {
                break;
            }
        }
    }
}

