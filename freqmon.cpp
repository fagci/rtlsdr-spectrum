#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include <liquid/liquid.h>
#include <rtl-sdr.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

// === Константы ===
constexpr size_t FFT_SIZE = 8192;
constexpr uint32_t SAMPLE_RATE = 2400000;
constexpr int GAIN = 439;
constexpr int WINDOW_WIDTH = 1024;
constexpr int WINDOW_HEIGHT = 768;
constexpr int SPECTRUM_HEIGHT = 200;
constexpr int WATERFALL_HEIGHT = 350;
constexpr int SCALE_HEIGHT = 50;
constexpr int PIP_WIDTH = 200;
constexpr int PIP_HEIGHT = 100;
constexpr float MIN_DB = -65.0f;
constexpr float MAX_DB = -15.0f;
constexpr double OVERLAP = 0.25;
constexpr double SCAN_START_FREQ = 0.0;
constexpr double SCAN_END_FREQ = 1'800'000'000.0;
constexpr double MIN_SPAN = 1e6;

constexpr double SCAN_STEP = SAMPLE_RATE * (1.0 - OVERLAP * 2.0);
constexpr size_t RING_BUFFER_SIZE = 16 * 1024 * 1024;

// === Band Plan ===
struct Band {
    std::string name;
    double start_hz;
    double end_hz;
};

std::vector<Band> bands = {
    {"LW", 100000.0, 300000.0},
    {"MW", 300000.0, 3000000.0},
    {"HF (3-30)", 3000000.0, 30000000.0},
    {"CB", 27000000.0, 27400000.0},
    {"11m", 25600000.0, 26100000.0},
    {"10m", 28000000.0, 29700000.0},
    {"VHF TV", 47000000.0, 86000000.0},
    {"FM Bcast", 87500000.0, 108000000.0},
    {"1.25m", 222000000.0, 225000000.0},
    {"2m", 144000000.0, 148000000.0},
    {"70cm", 420000000.0, 450000000.0},
    {"23cm", 1240000000.0, 1300000000.0}
};

// === RAII обёртки ===
struct FFTPlanWrapper {
  fftplan plan = nullptr;
  FFTPlanWrapper(size_t size, std::vector<liquid_float_complex> &in,
                 std::vector<liquid_float_complex> &out) {
    plan = fft_create_plan(size, in.data(), out.data(), LIQUID_FFT_FORWARD, 0);
  }
  ~FFTPlanWrapper() {
    if (plan)
      fft_destroy_plan(plan);
  }
};

struct RTLSDRDevice {
  rtlsdr_dev_t *dev = nullptr;
  ~RTLSDRDevice() {
    if (dev)
      rtlsdr_close(dev);
  }
};

// === Кольцевой буфер ===
class RingBuffer {
private:
  std::vector<uint8_t> buffer;
  std::atomic<size_t> read_pos{0};
  std::atomic<size_t> write_pos{0};
  size_t size;

public:
  RingBuffer(size_t size) : buffer(size), size(size) {}

  size_t available_read() const {
    size_t wp = write_pos.load(std::memory_order_acquire);
    size_t rp = read_pos.load(std::memory_order_acquire);
    if (wp >= rp)
      return wp - rp;
    return wp + size - rp;
  }

  size_t available_write() const { return size - available_read() - 1; }

  size_t write(const uint8_t *data, size_t len) {
    size_t available = available_write();
    if (len > available)
      len = available;

    size_t wp = write_pos.load(std::memory_order_relaxed);
    size_t to_end = size - wp;

    if (len <= to_end) {
      std::memcpy(&buffer[wp], data, len);
    } else {
      std::memcpy(&buffer[wp], data, to_end);
      std::memcpy(&buffer[0], data + to_end, len - to_end);
    }

    write_pos.store((wp + len) % size, std::memory_order_release);
    return len;
  }

  size_t read(uint8_t *data, size_t len) {
    size_t available = available_read();
    if (len > available)
      len = available;

    size_t rp = read_pos.load(std::memory_order_relaxed);
    size_t to_end = size - rp;

    if (len <= to_end) {
      std::memcpy(data, &buffer[rp], len);
    } else {
      std::memcpy(data, &buffer[rp], to_end);
      std::memcpy(data + to_end, &buffer[0], len - to_end);
    }

    read_pos.store((rp + len) % size, std::memory_order_release);
    return len;
  }
};

// === Глобальные переменные ===
std::atomic<bool> worker_running{true};
RingBuffer sample_buffer(RING_BUFFER_SIZE);

std::atomic<double> scan_start_freq{SCAN_START_FREQ};
std::atomic<double> scan_end_freq{SCAN_END_FREQ};
double display_start_freq = SCAN_START_FREQ;
double display_end_freq = SCAN_END_FREQ;

std::mutex freq_mutex;
std::atomic<double> current_center_freq{SCAN_START_FREQ};

std::atomic<bool> pip_mode{false};
std::atomic<double> pip_center_freq{0.0};
std::atomic<bool> full_main_complete{false};
std::atomic<bool> this_step_is_pip{false};

// === Вспомогательные функции ===
float u8_to_float(uint8_t v) { return ((float)v - 127.5f) / 128.0f; }

std::vector<float> make_hann_window(size_t N) {
  std::vector<float> w(N);
  for (size_t i = 0; i < N; ++i) {
    w[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * (float)i / (float)(N - 1)));
  }
  return w;
}

float lin_to_db(float x) { return 10.0f * std::log10(std::max(x, 1e-30f)); }

void get_classic_gradient(float t, Uint8 &r, Uint8 &g, Uint8 &b) {
  t = std::clamp(t, 0.0f, 1.0f);
  if (t < 0.2f) {
    float v = t / 0.2f;
    r = 0;
    g = 0;
    b = static_cast<Uint8>(v * 255);
  } else if (t < 0.4f) {
    float v = (t - 0.2f) / 0.2f;
    r = 0;
    g = static_cast<Uint8>(v * 255);
    b = 255;
  } else if (t < 0.6f) {
    float v = (t - 0.4f) / 0.2f;
    r = 0;
    g = 255;
    b = static_cast<Uint8>((1.0f - v) * 255);
  } else if (t < 0.8f) {
    float v = (t - 0.6f) / 0.2f;
    r = static_cast<Uint8>(v * 255);
    g = 255;
    b = 0;
  } else {
    float v = (t - 0.8f) / 0.2f;
    if (v < 0.5f) {
      r = 255;
      g = static_cast<Uint8>((1.0f - v * 2.0f) * 255);
      b = 0;
    } else {
      float w = (v - 0.5f) * 2.0f;
      r = 255;
      g = static_cast<Uint8>(w * 255);
      b = static_cast<Uint8>(w * 255);
    }
  }
}

void update_scan_range_from_display() {
  double new_scan_start = display_start_freq - SAMPLE_RATE * 0.6;
  double new_scan_end = display_end_freq + SAMPLE_RATE * 0.6;

  new_scan_start = std::max(SCAN_START_FREQ, new_scan_start);
  new_scan_end = std::min(SCAN_END_FREQ, new_scan_end);

  double min_scan_span = 10'000'000.0;
  if (new_scan_end - new_scan_start < min_scan_span) {
    double center = (new_scan_start + new_scan_end) / 2.0;
    new_scan_start = center - min_scan_span / 2;
    new_scan_end = center + min_scan_span / 2;

    new_scan_start = std::max(SCAN_START_FREQ, new_scan_start);
    new_scan_end = std::min(SCAN_END_FREQ, new_scan_end);
  }

  scan_start_freq.store(new_scan_start);
  scan_end_freq.store(new_scan_end);
}

// === Worker thread ===
void sdr_worker() {
  RTLSDRDevice sdr;
  if (rtlsdr_open(&sdr.dev, 0) < 0) {
    std::cerr << "Failed to open RTL-SDR device\n";
    return;
  }

  rtlsdr_set_sample_rate(sdr.dev, SAMPLE_RATE);
  rtlsdr_set_tuner_gain_mode(sdr.dev, 1);
  rtlsdr_set_tuner_gain(sdr.dev, GAIN);

  // Отключим AGC для более стабильного усиления
  rtlsdr_set_agc_mode(sdr.dev, 0);

  double main_center_freq = scan_start_freq.load();
  rtlsdr_set_center_freq(sdr.dev, static_cast<uint32_t>(main_center_freq));
  rtlsdr_reset_buffer(sdr.dev);
  usleep(10000); // Увеличенная пауза для стабилизации

  static bool alternate = false;

  while (worker_running.load()) {
    double current_scan_start = scan_start_freq.load();
    double current_scan_end = scan_end_freq.load();

    double next_center;
    bool do_pip_step = false;
    if (!pip_mode.load()) {
      next_center = main_center_freq + SCAN_STEP;
      if (next_center > current_scan_end || next_center < current_scan_start) {
        next_center = current_scan_start;
        full_main_complete.store(true);
      }
      main_center_freq = next_center;
      do_pip_step = false;
    } else {
      alternate = !alternate;
      if (alternate) {
        next_center = pip_center_freq.load();
        do_pip_step = true;
      } else {
        next_center = main_center_freq + SCAN_STEP;
        if (next_center > current_scan_end || next_center < current_scan_start) {
          next_center = current_scan_start;
          full_main_complete.store(true);
        }
        main_center_freq = next_center;
        do_pip_step = false;
      }
    }

    rtlsdr_set_center_freq(sdr.dev, static_cast<uint32_t>(next_center));
    current_center_freq.store(next_center);
    this_step_is_pip.store(do_pip_step);

    usleep(10000); // Увеличенная задержка после перестройки частоты

    std::vector<uint8_t> block(FFT_SIZE * 2);
    int total_read = 0;
    while (total_read < static_cast<int>(block.size())) {
      int n_read = 0;
      int ret = rtlsdr_read_sync(sdr.dev, block.data() + total_read,
                                 static_cast<int>(block.size()) - total_read, &n_read);
      if (ret < 0 || n_read <= 0) {
        usleep(1000);
        continue;
      }
      total_read += n_read;
    }

    size_t written = sample_buffer.write(block.data(), block.size());
    if (written < block.size()) {
      usleep(1000);
    }
  }
}

int main() {
  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    std::cerr << "SDL_Init failed: " << SDL_GetError() << "\n";
    return 1;
  }

  if (TTF_Init() < 0) {
    std::cerr << "TTF_Init failed: " << TTF_GetError() << "\n";
    SDL_Quit();
    return 1;
  }

  auto window_func = make_hann_window(FFT_SIZE);
  float win_sum_sq = 0.0f;
  for (float v : window_func)
    win_sum_sq += v * v;

  // Нормировочный коэффициент для FFT
  float fft_norm = 1.0f / (win_sum_sq * ((float)SAMPLE_RATE / FFT_SIZE));
  const double bin_width = static_cast<double>(SAMPLE_RATE) / FFT_SIZE;

  std::vector<liquid_float_complex> fft_in(FFT_SIZE);
  std::vector<liquid_float_complex> fft_out(FFT_SIZE);
  FFTPlanWrapper fft_wrapper(FFT_SIZE, fft_in, fft_out);
  if (!fft_wrapper.plan) {
    std::cerr << "Failed to create FFT plan\n";
    TTF_Quit();
    SDL_Quit();
    return 1;
  }

  std::thread worker_thread(sdr_worker);

  const double initial_scan_hz = SCAN_END_FREQ - SCAN_START_FREQ;
  size_t total_bins =
      static_cast<size_t>(std::ceil(initial_scan_hz / bin_width));

  std::vector<float> cycle_psd_sum(total_bins, 0.0f);
  std::vector<int> cycle_count(total_bins, 0);
  std::vector<float> display_scan(total_bins, MIN_DB - 100.0f);

  SDL_Window *sdl_window = SDL_CreateWindow(
      "Spectrum Analyzer (Wide Band with Persistence)", SDL_WINDOWPOS_UNDEFINED,
      SDL_WINDOWPOS_UNDEFINED, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
  if (!sdl_window) {
    std::cerr << "Window creation failed\n";
    worker_running = false;
    worker_thread.join();
    TTF_Quit();
    SDL_Quit();
    return 1;
  }

  SDL_Renderer *renderer =
      SDL_CreateRenderer(sdl_window, -1, SDL_RENDERER_ACCELERATED);
  if (!renderer) {
    std::cerr << "Renderer creation failed\n";
    SDL_DestroyWindow(sdl_window);
    worker_running = false;
    worker_thread.join();
    TTF_Quit();
    SDL_Quit();
    return 1;
  }

  TTF_Font *font =
      TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12);
  if (!font) {
    font = TTF_OpenFont("/System/Library/Fonts/SFNS.ttf", 12);
  }
  if (!font) {
    font = TTF_OpenFont("C:\\Windows\\Fonts\\arial.ttf", 12);
  }
  if (!font) {
    std::cerr << "Warning: Font not found\n";
  }

  SDL_Texture *waterfall_tex = SDL_CreateTexture(
      renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING,
      WINDOW_WIDTH, WATERFALL_HEIGHT);

  std::vector<std::vector<Uint32>> waterfall_lines(
      WATERFALL_HEIGHT, std::vector<Uint32>(WINDOW_WIDTH, 0));
  int waterfall_top_line = 0;

  SDL_Texture *pip_tex = SDL_CreateTexture(
      renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING,
      PIP_WIDTH, PIP_HEIGHT);

  std::vector<std::vector<Uint32>> pip_waterfall_lines(
      PIP_HEIGHT, std::vector<Uint32>(PIP_WIDTH, 0));
  int pip_top_line = 0;

  bool quit = false;
  int mouse_x = -1, mouse_y = -1;
  bool mouse_dragging = false;
  int drag_start_x = 0;
  double drag_start_start_freq = 0;
  double drag_start_end_freq = 0;

  const double MAX_SPAN = SCAN_END_FREQ - SCAN_START_FREQ;

  std::vector<uint8_t> rbuf(FFT_SIZE * 2);

  update_scan_range_from_display();

  double prev_scan_start = SCAN_START_FREQ;
  double prev_scan_end = SCAN_END_FREQ;
  bool prev_pip_mode = false;

  while (!quit) {
    size_t available = sample_buffer.available_read();
    if (available < rbuf.size()) {
      SDL_Delay(1);
      continue;
    }

    sample_buffer.read(rbuf.data(), rbuf.size());

    double center_freq = current_center_freq.load();
    double current_scan_start = scan_start_freq.load();
    double current_scan_end = scan_end_freq.load();
    bool current_pip_mode = pip_mode.load();

    if (current_pip_mode != prev_pip_mode || current_scan_start != prev_scan_start || current_scan_end != prev_scan_end) {
      std::fill(cycle_psd_sum.begin(), cycle_psd_sum.end(), 0.0f);
      std::fill(cycle_count.begin(), cycle_count.end(), 0);
      prev_scan_start = current_scan_start;
      prev_scan_end = current_scan_end;
      prev_pip_mode = current_pip_mode;
    }

    double display_total_hz = display_end_freq - display_start_freq;

    for (size_t i = 0; i < FFT_SIZE; ++i) {
      float I = u8_to_float(rbuf[2 * i]);
      float Q = u8_to_float(rbuf[2 * i + 1]);
      fft_in[i] = liquid_float_complex{I * window_func[i], Q * window_func[i]};
    }

    fft_execute(fft_wrapper.plan);
    fft_shift(fft_out.data(), FFT_SIZE);

    bool is_pip_step = this_step_is_pip.load();

    if (is_pip_step) {
      // PiP processing
      std::vector<Uint32> pip_line(PIP_WIDTH, 0);
      double pip_start = pip_center_freq.load() - SAMPLE_RATE / 2.0;
      double pip_bin_width = static_cast<double>(SAMPLE_RATE) / FFT_SIZE;
      for (int px = 0; px < PIP_WIDTH; ++px) {
        double rel = static_cast<double>(px) / (PIP_WIDTH - 1.0);
        double f = pip_start + rel * SAMPLE_RATE;
        size_t k = static_cast<size_t>(std::round((f - pip_start) / pip_bin_width));
        if (k >= FFT_SIZE) k = FFT_SIZE - 1;
        float re = fft_out[k].real;
        float im = fft_out[k].imag;
        float mag = re * re + im * im;
        float psd = mag * fft_norm;
        float db_val = lin_to_db(psd);
        float norm = std::clamp((db_val - MIN_DB) / (MAX_DB - MIN_DB), 0.0f, 1.0f);
        Uint8 r, g, b;
        get_classic_gradient(norm, r, g, b);
        pip_line[px] = (255U << 24) | (r << 16) | (g << 8) | b;
      }

      pip_top_line = (pip_top_line - 1 + PIP_HEIGHT) % PIP_HEIGHT;
      pip_waterfall_lines[pip_top_line] = pip_line;

      if (pip_tex) {
        void *pixels;
        int pitch;
        if (SDL_LockTexture(pip_tex, nullptr, &pixels, &pitch) == 0) {
          for (int y = 0; y < PIP_HEIGHT; ++y) {
            int src_y = (pip_top_line + y) % PIP_HEIGHT;
            Uint8 *row = static_cast<Uint8 *>(pixels) + y * pitch;
            std::memcpy(row, pip_waterfall_lines[src_y].data(),
                        PIP_WIDTH * sizeof(Uint32));
          }
          SDL_UnlockTexture(pip_tex);
        }
      }
    } else {
      // Main processing
      for (size_t k = 0; k < FFT_SIZE; ++k) {
        float re = fft_out[k].real;
        float im = fft_out[k].imag;
        float psd = (re * re + im * im) / (win_sum_sq * bin_width + 1e-30f);

        double bin_freq = center_freq + (k - FFT_SIZE / 2.0) * bin_width;
        if (bin_freq >= SCAN_START_FREQ && bin_freq < SCAN_END_FREQ) {
          size_t global_idx =
              static_cast<size_t>(std::round((bin_freq - SCAN_START_FREQ) / bin_width));
          if (global_idx < total_bins) {
            cycle_psd_sum[global_idx] += psd;
            cycle_count[global_idx]++;
          }
        }
      }
    }

    // === Рендеринг ===
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);

    // Сетка с динамическим шагом для видимости меток при большом зуме
    SDL_SetRenderDrawColor(renderer, 64, 64, 64, 255);
    double display_total_mhz = display_total_hz / 1e6;
    double start_mhz = display_start_freq / 1e6;
    double end_mhz = display_end_freq / 1e6;

    double grid_step_mhz = 10.0;
    if (display_total_mhz > 500) grid_step_mhz = 100.0;
    else if (display_total_mhz > 100) grid_step_mhz = 50.0;
    else if (display_total_mhz < 10) grid_step_mhz = 1.0;
    else if (display_total_mhz < 1) grid_step_mhz = 0.1;

    double start_rounded = std::ceil(start_mhz / grid_step_mhz) * grid_step_mhz;
    double end_rounded = std::floor(end_mhz / grid_step_mhz) * grid_step_mhz;

    for (double mhz = start_rounded; mhz <= end_rounded; mhz += grid_step_mhz) {
      double rel_pos = (mhz - start_mhz) / display_total_mhz;
      int x = static_cast<int>(rel_pos * WINDOW_WIDTH);
      if (x >= 0 && x < WINDOW_WIDTH) {
        SDL_RenderDrawLine(renderer, x, 0, x,
                           SPECTRUM_HEIGHT + WATERFALL_HEIGHT);
      }
    }

    // Main Waterfall update
    if (full_main_complete.load()) {
      full_main_complete.store(false);
      // Compute average PSD and dB
      std::vector<float> cycle_scan_avg(total_bins, MIN_DB - 100.0f);
      for (size_t i = 0; i < total_bins; ++i) {
        if (cycle_count[i] > 0) {
          float avg_psd = cycle_psd_sum[i] / cycle_count[i];
          cycle_scan_avg[i] = lin_to_db(avg_psd);
          display_scan[i] = cycle_scan_avg[i];
        }
      }

      // Reset accumulators for next scan
      std::fill(cycle_psd_sum.begin(), cycle_psd_sum.end(), 0.0f);
      std::fill(cycle_count.begin(), cycle_count.end(), 0);

      // Waterfall линия
      std::vector<Uint32> waterfall_line(WINDOW_WIDTH, 0);
      const float db_range = MAX_DB - MIN_DB;
      for (int x = 0; x < WINDOW_WIDTH; ++x) {
        double freq_low =
            display_start_freq + (x / (double)WINDOW_WIDTH) * display_total_hz;
        double freq_high =
            display_start_freq + ((x + 1.0) / (double)WINDOW_WIDTH) * display_total_hz;

        size_t low_idx = static_cast<size_t>(std::floor((freq_low - SCAN_START_FREQ) / bin_width));
        size_t high_idx = static_cast<size_t>(std::ceil((freq_high - SCAN_START_FREQ) / bin_width));

        float db_val = MIN_DB - 100.0f;
        for (size_t idx = low_idx; idx < high_idx && idx < total_bins; ++idx) {
          db_val = std::max(db_val, cycle_scan_avg[idx]);
        }

        float norm = std::clamp((db_val - MIN_DB) / db_range, 0.0f, 1.0f);
        Uint8 r, g, b;
        get_classic_gradient(norm, r, g, b);
        waterfall_line[x] = (255U << 24) | (r << 16) | (g << 8) | b;
      }

      waterfall_top_line =
          (waterfall_top_line - 1 + WATERFALL_HEIGHT) % WATERFALL_HEIGHT;
      waterfall_lines[waterfall_top_line] = waterfall_line;

      if (waterfall_tex) {
        void *pixels;
        int pitch;
        if (SDL_LockTexture(waterfall_tex, nullptr, &pixels, &pitch) == 0) {
          for (int y = 0; y < WATERFALL_HEIGHT; ++y) {
            int src_y = (waterfall_top_line + y) % WATERFALL_HEIGHT;
            Uint8 *row = static_cast<Uint8 *>(pixels) + y * pitch;
            std::memcpy(row, waterfall_lines[src_y].data(),
                        WINDOW_WIDTH * sizeof(Uint32));
          }
          SDL_UnlockTexture(waterfall_tex);
        }
      }
    }

    // Спектр
    const float db_range = MAX_DB - MIN_DB;
    float prev_y = SPECTRUM_HEIGHT;

    for (int x = 0; x < WINDOW_WIDTH; ++x) {
      double freq_low =
          display_start_freq + (x / (double)WINDOW_WIDTH) * display_total_hz;
      double freq_high =
          display_start_freq + ((x + 1.0) / (double)WINDOW_WIDTH) * display_total_hz;

      size_t low_idx = static_cast<size_t>(std::floor((freq_low - SCAN_START_FREQ) / bin_width));
      size_t high_idx = static_cast<size_t>(std::ceil((freq_high - SCAN_START_FREQ) / bin_width));

      float db_val = MIN_DB - 100.0f;
      for (size_t idx = low_idx; idx < high_idx && idx < total_bins; ++idx) {
        db_val = std::max(db_val, display_scan[idx]);
      }

      float norm = std::clamp((db_val - MIN_DB) / db_range, 0.0f, 1.0f);
      float y = SPECTRUM_HEIGHT - (norm * SPECTRUM_HEIGHT);

      SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);  // Белый цвет для спектра
      if (x > 0) {
        SDL_RenderDrawLine(renderer, x - 1, (int)prev_y, x, (int)y);
      }
      prev_y = y;
    }

    if (waterfall_tex) {
      SDL_Rect waterfall_rect = {0, SPECTRUM_HEIGHT, WINDOW_WIDTH,
                                 WATERFALL_HEIGHT};
      SDL_RenderCopy(renderer, waterfall_tex, nullptr, &waterfall_rect);
    }

    // PiP
    if (pip_mode.load() && pip_tex) {
      SDL_Rect pip_rect = {WINDOW_WIDTH - PIP_WIDTH, 0, PIP_WIDTH, PIP_HEIGHT};
      SDL_RenderCopy(renderer, pip_tex, nullptr, &pip_rect);

      if (font) {
        char pip_text[16];
        snprintf(pip_text, sizeof(pip_text), "PiP %.1f MHz", pip_center_freq.load() / 1e6);
        SDL_Surface *text_surface = TTF_RenderText_Solid(font, pip_text, {255, 255, 255, 255});
        if (text_surface) {
          SDL_Texture *text_texture = SDL_CreateTextureFromSurface(renderer, text_surface);
          if (text_texture) {
            int tw, th;
            SDL_QueryTexture(text_texture, nullptr, nullptr, &tw, &th);
            SDL_Rect dst = {WINDOW_WIDTH - PIP_WIDTH + 5, PIP_HEIGHT + 5, tw, th};
            SDL_RenderCopy(renderer, text_texture, nullptr, &dst);
            SDL_DestroyTexture(text_texture);
          }
          SDL_FreeSurface(text_surface);
        }
      }
    }

    // Шкала
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_Rect scale_rect = {0, WINDOW_HEIGHT - SCALE_HEIGHT, WINDOW_WIDTH,
                           SCALE_HEIGHT};
    SDL_RenderFillRect(renderer, &scale_rect);

    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    if (font) {
      for (double mhz = start_rounded; mhz <= end_rounded; mhz += grid_step_mhz) {
        double rel_pos = (mhz - start_mhz) / display_total_mhz;
        int x = static_cast<int>(rel_pos * WINDOW_WIDTH);
        if (x < 10 || x > WINDOW_WIDTH - 10)
          continue;

        char freq_text[32];
        if (grid_step_mhz < 1.0) {
          snprintf(freq_text, sizeof(freq_text), "%.1f", mhz);
        } else {
          snprintf(freq_text, sizeof(freq_text), "%.0f", mhz);
        }
        SDL_Surface *text_surface =
            TTF_RenderText_Solid(font, freq_text, {0, 0, 0, 255});
        if (text_surface) {
          SDL_Texture *text_texture =
              SDL_CreateTextureFromSurface(renderer, text_surface);
          if (text_texture) {
            int tw, th;
            SDL_QueryTexture(text_texture, nullptr, nullptr, &tw, &th);
            SDL_Rect dst = {x - tw / 2, WINDOW_HEIGHT - SCALE_HEIGHT + 10, tw,
                            th};
            SDL_RenderCopy(renderer, text_texture, nullptr, &dst);
            SDL_DestroyTexture(text_texture);
          }
          SDL_FreeSurface(text_surface);
        }
      }
    }

    // Band Plan labels
    if (font) {
      for (const auto& band : bands) {
        double b_start = band.start_hz / 1e6;
        double b_end = band.end_hz / 1e6;
        if (b_end < start_mhz || b_start > end_mhz) continue;
        double inter_start = std::max(b_start, start_mhz);
        double inter_end = std::min(b_end, end_mhz);
        if (inter_end <= inter_start) continue;
        double inter_width_mhz = inter_end - inter_start;
        double inter_center = (inter_start + inter_end) / 2.0;
        double rel_pos = (inter_center - start_mhz) / display_total_mhz;
        int x = static_cast<int>(rel_pos * WINDOW_WIDTH);
        if (x < 0 || x >= WINDOW_WIDTH) continue;
        // Only label if wide enough (>20 pixels)
        double px_width = inter_width_mhz / display_total_mhz * WINDOW_WIDTH;
        if (px_width < 20.0) continue;
        SDL_Surface *text_surface = TTF_RenderText_Solid(font, band.name.c_str(), {0, 0, 0, 255});
        if (text_surface) {
          int tw = text_surface->w;
          int th = text_surface->h;
          if (x - tw / 2 >= 0 && x + tw / 2 <= WINDOW_WIDTH) {
            SDL_Texture *text_texture = SDL_CreateTextureFromSurface(renderer, text_surface);
            if (text_texture) {
              SDL_Rect dst = {x - tw / 2, WINDOW_HEIGHT - SCALE_HEIGHT + (SCALE_HEIGHT - th) / 2, tw, th};
              SDL_RenderCopy(renderer, text_texture, nullptr, &dst);
              SDL_DestroyTexture(text_texture);
            }
          }
          SDL_FreeSurface(text_surface);
        }
      }
    }

    // Tooltip
    SDL_GetMouseState(&mouse_x, &mouse_y);
    if (mouse_x >= 0 && mouse_x < WINDOW_WIDTH && mouse_y >= 0 &&
        mouse_y < (WINDOW_HEIGHT - SCALE_HEIGHT)) {
      double freq =
          display_start_freq +
          (static_cast<double>(mouse_x) / WINDOW_WIDTH) * display_total_hz;
      char freq_text[32];
      snprintf(freq_text, sizeof(freq_text), "%.3f MHz", freq / 1e6);
      if (font) {
        SDL_Surface *text_surface =
            TTF_RenderText_Solid(font, freq_text, {255, 255, 255, 255});
        if (text_surface) {
          SDL_Texture *text_texture =
              SDL_CreateTextureFromSurface(renderer, text_surface);
          if (text_texture) {
            int tw, th;
            SDL_QueryTexture(text_texture, nullptr, nullptr, &tw, &th);
            SDL_SetRenderDrawColor(renderer, 0, 0, 0, 200);
            SDL_Rect bg = {mouse_x + 10, mouse_y + 10, tw + 4, th + 4};
            SDL_RenderFillRect(renderer, &bg);
            SDL_Rect text_dst = {mouse_x + 12, mouse_y + 12, tw, th};
            SDL_RenderCopy(renderer, text_texture, nullptr, &text_dst);
            SDL_DestroyTexture(text_texture);
          }
          SDL_FreeSurface(text_surface);
        }
      }
    }

    // Отладочная информация
    if (font) {
      char info_text[256];
      snprintf(info_text, sizeof(info_text),
               "Display: %.1f-%.1f MHz | Scan: %.1f-%.1f MHz | Center: %.1f "
               "MHz | Gain: %d | PiP: %s",
               display_start_freq / 1e6, display_end_freq / 1e6,
               current_scan_start / 1e6, current_scan_end / 1e6,
               center_freq / 1e6, GAIN, pip_mode.load() ? "ON" : "OFF");
      SDL_Surface *info_surface =
          TTF_RenderText_Solid(font, info_text, {255, 255, 0, 255});
      if (info_surface) {
        SDL_Texture *info_texture =
            SDL_CreateTextureFromSurface(renderer, info_surface);
        if (info_texture) {
          SDL_Rect dst = {10, 10, info_surface->w, info_surface->h};
          SDL_RenderCopy(renderer, info_texture, nullptr, &dst);
          SDL_DestroyTexture(info_texture);
        }
        SDL_FreeSurface(info_surface);
      }
    }

    SDL_RenderPresent(renderer);

    // Обработка событий
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
      if (e.type == SDL_QUIT) {
        quit = true;
      } else if (e.type == SDL_MOUSEMOTION) {
        mouse_x = e.motion.x;
        mouse_y = e.motion.y;

        if (mouse_dragging) {
          int delta_x = mouse_x - drag_start_x;
          double delta_freq =
              (delta_x / (double)WINDOW_WIDTH) * display_total_hz;

          display_start_freq = drag_start_start_freq - delta_freq;
          display_end_freq = drag_start_end_freq - delta_freq;

          display_start_freq = std::max(SCAN_START_FREQ, display_start_freq);
          display_end_freq = std::min(SCAN_END_FREQ, display_end_freq);

          double span = display_end_freq - display_start_freq;
          if (span < MIN_SPAN) {
            double center = (display_start_freq + display_end_freq) / 2.0;
            display_start_freq = center - MIN_SPAN / 2;
            display_end_freq = center + MIN_SPAN / 2;
          }

          update_scan_range_from_display();
        }
      } else if (e.type == SDL_MOUSEBUTTONDOWN) {
        if (e.button.button == SDL_BUTTON_RIGHT) {
          mouse_dragging = true;
          drag_start_x = e.button.x;
          drag_start_start_freq = display_start_freq;
          drag_start_end_freq = display_end_freq;
        } else if (e.button.button == SDL_BUTTON_LEFT) {
          if (e.button.y > WINDOW_HEIGHT - SCALE_HEIGHT) {
            // Click in scale area - select band
            double mhz = start_mhz + (static_cast<double>(e.button.x) / WINDOW_WIDTH) * display_total_mhz;
            double freq = mhz * 1e6;
            for (const auto& band : bands) {
              if (freq >= band.start_hz && freq <= band.end_hz) {
                display_start_freq = band.start_hz;
                display_end_freq = band.end_hz;
                display_start_freq = std::max(SCAN_START_FREQ, display_start_freq);
                display_end_freq = std::min(SCAN_END_FREQ, display_end_freq);
                double span = display_end_freq - display_start_freq;
                if (span < MIN_SPAN) {
                  double center = (display_start_freq + display_end_freq) / 2.0;
                  display_start_freq = center - MIN_SPAN / 2;
                  display_end_freq = center + MIN_SPAN / 2;
                }
                update_scan_range_from_display();
                break;
              }
            }
          } else if (e.button.y < SPECTRUM_HEIGHT + WATERFALL_HEIGHT) {
            // Click in spectrum/waterfall - quick view (zoom to 1 MHz around freq)
            double freq = display_start_freq + (static_cast<double>(e.button.x) / WINDOW_WIDTH) * display_total_hz;
            double new_span = 1e6;  // 1 MHz quick view
            display_start_freq = freq - new_span / 2.0;
            display_end_freq = freq + new_span / 2.0;
            display_start_freq = std::max(SCAN_START_FREQ, display_start_freq);
            display_end_freq = std::min(SCAN_END_FREQ, display_end_freq);
            if (display_end_freq - display_start_freq < new_span) {
              double center = (display_start_freq + display_end_freq) / 2.0;
              display_start_freq = center - new_span / 2.0;
              display_end_freq = center + new_span / 2.0;
              display_start_freq = std::max(SCAN_START_FREQ, display_start_freq);
              display_end_freq = std::min(SCAN_END_FREQ, display_end_freq);
            }
            update_scan_range_from_display();
          }
        }
      } else if (e.type == SDL_MOUSEBUTTONUP) {
        if (e.button.button == SDL_BUTTON_RIGHT) {
          mouse_dragging = false;
        }
      } else if (e.type == SDL_MOUSEWHEEL) {
        double cursor_freq =
            display_start_freq +
            (static_cast<double>(mouse_x) / WINDOW_WIDTH) * display_total_hz;

        double current_span = display_end_freq - display_start_freq;
        double zoom_factor = (e.wheel.y > 0) ? 0.7 : 1.3;

        double new_span = current_span * zoom_factor;
        new_span = std::clamp(new_span, MIN_SPAN, MAX_SPAN);

        double cursor_ratio = (cursor_freq - display_start_freq) / current_span;
        display_start_freq = cursor_freq - cursor_ratio * new_span;
        display_end_freq = display_start_freq + new_span;

        if (display_start_freq < SCAN_START_FREQ) {
          display_start_freq = SCAN_START_FREQ;
          display_end_freq = SCAN_START_FREQ + new_span;
        }
        if (display_end_freq > SCAN_END_FREQ) {
          display_end_freq = SCAN_END_FREQ;
          display_start_freq = SCAN_END_FREQ - new_span;
        }

        update_scan_range_from_display();
      } else if (e.type == SDL_KEYDOWN) {
        double scroll_amount = display_total_hz * 0.1;

        if (e.key.keysym.sym == SDLK_LEFT) {
          display_start_freq -= scroll_amount;
          display_end_freq -= scroll_amount;

          if (display_start_freq < SCAN_START_FREQ) {
            double diff = SCAN_START_FREQ - display_start_freq;
            display_start_freq = SCAN_START_FREQ;
            display_end_freq += diff;
          }

          update_scan_range_from_display();
        } else if (e.key.keysym.sym == SDLK_RIGHT) {
          display_start_freq += scroll_amount;
          display_end_freq += scroll_amount;

          if (display_end_freq > SCAN_END_FREQ) {
            double diff = display_end_freq - SCAN_END_FREQ;
            display_end_freq = SCAN_END_FREQ;
            display_start_freq -= diff;
          }

          update_scan_range_from_display();
        } else if (e.key.keysym.sym == SDLK_HOME) {
          display_start_freq = SCAN_START_FREQ;
          display_end_freq = SCAN_END_FREQ;
          update_scan_range_from_display();
        } else if (e.key.keysym.sym == SDLK_p) {
          bool new_mode = !pip_mode.load();
          pip_mode.store(new_mode);
          if (new_mode) {
            double mouse_freq =
                display_start_freq +
                (static_cast<double>(mouse_x) / WINDOW_WIDTH) * display_total_hz;
            pip_center_freq.store(mouse_freq);
            // Clear PiP waterfall
            for (auto& line : pip_waterfall_lines) {
              std::fill(line.begin(), line.end(), 0U);
            }
            pip_top_line = 0;
          }
        }
      }
    }
  }

  worker_running = false;
  worker_thread.join();

  if (pip_tex)
    SDL_DestroyTexture(pip_tex);
  if (waterfall_tex)
    SDL_DestroyTexture(waterfall_tex);
  if (font)
    TTF_CloseFont(font);
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(sdl_window);
  TTF_Quit();
  SDL_Quit();

  return 0;
}
