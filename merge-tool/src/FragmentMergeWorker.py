import os.path
from typing import List
import wave
import ffmpeg
from moviepy import VideoFileClip, AudioFileClip

from src.utils import only_files, create_folder
from datetime import datetime

class FragmentMergeWorker:
  def __init__(self, duration: int, in_path: str, out_path: str = None):
    self.duration = duration
    self.in_path = in_path
    self.out_path = out_path

  def _split_by_type(self, files: List[str]) -> tuple[list[str | None], list[str | None]]:
    wav_files = [f for f in files if f.endswith('.wav') and 'emg' not in f]
    mkv_files = [f for f in files if f.endswith('.mp4')]
    return wav_files, mkv_files

  def _combine_by_duration(self, timespans: List[datetime] = None):
    chunk_time = self.duration * 60
    chunks = [[]]
    start_time = timespans[0]
    index = 0
    for timespan in timespans:
      diff = timespan - start_time
      if diff.seconds > chunk_time:
        start_time = timespan
        index += 1
        chunks.append([])
      chunks[index].append(timespan)
    return chunks

  def _merge_wav(self, in_files:List[str], outfile:str):
    data = []
    for infile in in_files:
      try:
        w = wave.open(infile, 'rb')
        data.append([w.getparams(), w.readframes(w.getnframes())])
        w.close()
      except EOFError as exc:
        print('error reading:', infile)

    output = wave.open(outfile, 'wb')
    output.setparams(data[0][0])
    for i in range(len(data)):
      output.writeframes(data[i][1])
    print('--> write:', outfile)
    output.close()

  def _merge_mkv(self,in_files:List[str], outfile:str):
    print('--> write:', outfile)
    video_files = [ffmpeg.input(f)  for f in in_files]
    print('video_files-->', video_files)
    concatenated_streams = ffmpeg.concat(*video_files, v=1)
    ffmpeg.output(concatenated_streams, outfile).run()


  def combine_audio(self, in_path: str, out_path: str, fps=25):
    vide_clip = VideoFileClip(f'{in_path}.mp4')
    audio_background = AudioFileClip(f'{in_path}.wav')
    vide_clip.audio = audio_background
    vide_clip.write_videofile(f'{out_path}.mp4', fps=fps)

  def _timespan_to_file_name(self, timespan: datetime) -> str:
    return timespan.strftime("%Y-%m-%d_%H.%M.%S")

  def form_file_paths(self, chunk: List[datetime]) -> tuple[list[str], str]:
    diff = chunk[-1] - chunk[0]


    in_files = [os.path.join(self.in_path, self._timespan_to_file_name(f)) for f in chunk]
    outfile = os.path.join(self.out_path, f'{self._timespan_to_file_name(chunk[0])}_{str(round(diff.seconds / 60, 2))}')
    return in_files, outfile

  def _merge_audio_chunks(self, chunk: List[datetime]):
    in_files, outfile = self.form_file_paths(chunk)
    self._merge_wav([f+ '.wav' for f in in_files], outfile + '.wav')

  def _merge_video_chunks(self, chunk: List[datetime]):
    in_files, outfile = self.form_file_paths(chunk)
    self._merge_mkv([f + '.mp4' for f in in_files], outfile + '.mp4')

  def merge(self):
    print('start merging...', self.in_path)
    all_files = only_files(self.in_path)
    files_with_size = [f for f in all_files if os.path.getsize(os.path.join(self.in_path, f)) > 0]
    wav_files, mkv_files = self._split_by_type(files_with_size)
    print('--> create:', self.out_path)
    # 1. merge audio files
    audio_timespans = [datetime.strptime(f.replace('.wav', ''), '%Y-%m-%d_%H.%M.%S') for f in wav_files]
    audio_timespans.sort()
    chunks_audio = self._combine_by_duration(timespans=audio_timespans)
    print('chunks_audio:', [len(chunk) for chunk in chunks_audio])

    for chunk in chunks_audio:
      self._merge_audio_chunks(chunk)

    # 2. merge video files
    video_timespans = [datetime.strptime(f.replace('.mp4', ''), '%Y-%m-%d_%H.%M.%S') for f in mkv_files]
    video_timespans.sort()
    chunks_video = self._combine_by_duration(timespans=video_timespans)
    print('chunks_video:', [len(chunk) for chunk in chunks_video])
    for chunk in chunks_video:
      self._merge_video_chunks(chunk)

    # for i, chunk in enumerate(chunks_audio):
    #   in_audio, _ = self.form_file_paths(chunk)
    #   in_video, __ = self.form_file_paths(chunks_video[i])
    #   diff_a = chunk[-1] - chunk[0]
    #   diff_v = chunks_video[i][-1] - chunks_video[i][0]
    #   print('audio duration:', diff_a, 'seconds')
    #   print('video duration:', diff_v, 'seconds')
    #   for j, f in enumerate(in_audio):
    #     v = ''
    #     try:
    #       v = in_video[j]
    #     except IndexError:
    #       continue
    #     print(f + ' - ' + v)
    print('end merging...', self.in_path)
