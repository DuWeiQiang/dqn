#include <cmath>
#include <iostream>
#include <ale_interface.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include "prettyprint.hpp"
#include "dqn.hpp"
#include <boost/filesystem.hpp>
#include <thread>
#include <mutex>
#include <algorithm>
#include <chrono>
#include <limits>

using namespace boost::filesystem;

// DQN Parameters
DEFINE_bool(gpu, true, "Use GPU to brew Caffe");
DEFINE_int32(device, -1, "Which GPU to use");
DEFINE_bool(gui, false, "Open a GUI window");
DEFINE_string(save, "", "Prefix for saving snapshots");
DEFINE_string(rom, "", "Atari 2600 ROM file to play");
DEFINE_int32(memory, 400000, "Capacity of replay memory");
DEFINE_int32(explore, 1000000, "Iterations for epsilon to reach given value.");
DEFINE_double(epsilon, .1, "Value of epsilon after explore iterations.");
DEFINE_double(gamma, .99, "Discount factor of future rewards (0,1]");
DEFINE_int32(clone_freq, 10000, "Frequency (steps) of cloning the target network.");
DEFINE_int32(memory_threshold, 50000, "Number of transitions to start learning");
DEFINE_int32(skip_frame, 3, "Number of frames skipped");
DEFINE_string(save_screen, "", "File prefix in to save frames");
DEFINE_string(save_binary_screen, "", "File prefix in to save binary frames");
DEFINE_string(weights, "", "The pretrained weights load (*.caffemodel).");
DEFINE_string(snapshot, "", "The solver state to load (*.solverstate).");
DEFINE_bool(evaluate, false, "Evaluation mode: only playing a game, no updates");
DEFINE_double(evaluate_with_epsilon, .05, "Epsilon value to be used in evaluation mode");
DEFINE_int32(evaluate_freq, 250000, "Frequency (steps) between evaluations");
DEFINE_int32(repeat_games, 32, "Number of games played in evaluation mode");
DEFINE_string(solver, "recurrent_solver.prototxt", "Solver parameter file (*.prototxt)");

double CalculateEpsilon(const int iter) {
  if (iter < FLAGS_explore) {
    return 1.0 - (1.0 - FLAGS_epsilon) * (static_cast<double>(iter) / FLAGS_explore);
  } else {
    return FLAGS_epsilon;
  }
}

void SaveScreen(const ALEScreen& screen, const ALEInterface& ale,
                const string filename) {
  IntMatrix screen_matrix;
  for (auto row = 0; row < screen.height(); row++) {
    IntVect row_vec;
    for (auto col = 0; col < screen.width(); col++) {
      int pixel = screen.get(row, col);
      row_vec.emplace_back(pixel);
    }
    screen_matrix.emplace_back(row_vec);
  }
  ale.theOSystem->p_export_screen->save_png(&screen_matrix, filename);
}

void SaveInputFrame(const dqn::FrameData& frame, const string filename) {
  std::ofstream ofs;
  ofs.open(filename, ios::out | ios::binary);
  for (int i = 0; i < dqn::kCroppedFrameDataSize; ++i) {
    ofs.write((char*) &frame[i], sizeof(uint8_t));
  }
  ofs.close();
}

void InitializeALE(ALEInterface& ale, bool display_screen, std::string& rom) {
  ale.set("display_screen", display_screen);
  ale.set("disable_color_averaging", true);
  ale.loadROM(rom);
}

/**
 * Play one episode and return the total score
 */
double PlayOneEpisode(ALEInterface& ale, dqn::DQN& dqn, const double epsilon,
                      const bool update) {
  CHECK(!ale.game_over());
  std::deque<dqn::FrameDataSp> past_frames;
  dqn::Episode episode;
  auto total_score = 0.0;
  for (auto frame = 0; !ale.game_over(); ++frame) {
    const ALEScreen& screen = ale.getScreen();
    if (!FLAGS_save_screen.empty()) {
      std::stringstream ss;
      ss << FLAGS_save_screen << setfill('0') << setw(5) <<
          std::to_string(frame) << ".png";
      SaveScreen(screen, ale, ss.str());
    }
    const dqn::FrameDataSp current_frame = dqn::PreprocessScreen(screen);
    past_frames.push_back(current_frame);
    if (!FLAGS_save_binary_screen.empty()) {
      static int binary_save_num = 0;
      string fname = FLAGS_save_binary_screen +
          std::to_string(binary_save_num++) + ".bin";
      SaveInputFrame(*current_frame, fname);
    }
    if (past_frames.size() < dqn::kInputFramesPerTimestep) {
      // If there are not past frames enough for DQN input, just select NOOP
      for (int i = 0; i < FLAGS_skip_frame + 1 && !ale.game_over(); ++i) {
        total_score += ale.act(PLAYER_A_NOOP);
      }
      continue;
    }
    while (past_frames.size() > dqn::kInputFramesPerTimestep) {
      past_frames.pop_front();
    }
    CHECK_EQ(past_frames.size(), dqn::kInputFramesPerTimestep);
    dqn::InputFrames input_frames;
    std::copy(past_frames.begin(), past_frames.end(), input_frames.begin());
    const auto action = dqn.SelectAction(input_frames, epsilon, frame > 0);
    auto immediate_score = 0.0;
    for (auto i = 0; i < FLAGS_skip_frame + 1 && !ale.game_over(); ++i) {
      immediate_score += ale.act(action);
    }
    total_score += immediate_score;
    // Rewards for DQN are normalized as follows:
    // 1 for any positive score, -1 for any negative score, otherwise 0
    const auto reward = immediate_score == 0 ? 0 : immediate_score /
        std::abs(immediate_score);
    assert(reward <= 1 && reward >= -1);
    if (update) {
      // Add the current transition to replay memory
      const auto transition = ale.game_over() ?
          dqn::Transition(current_frame, action, reward, boost::none) :
          dqn::Transition(current_frame, action, reward,
                          dqn::PreprocessScreen(ale.getScreen()));
      episode.push_back(transition);
      if (dqn.memory_size() > FLAGS_memory_threshold) {
        dqn.UpdateRandom();
      }
    }
  }
  if (update) {
    dqn.RememberEpisode(episode);
  }
  ale.reset_game();
  return total_score;
}

/**
 * Evaluate the current player
 */
double Evaluate(ALEInterface& ale, dqn::DQN& dqn) {
  // std::vector<double> scores = PlayParallelEpisodes(
  //     dqn, FLAGS_evaluate_with_epsilon, false);
  std::vector<double> scores;
  scores.push_back(PlayOneEpisode(ale, dqn, FLAGS_evaluate_with_epsilon, false));
  double total_score = 0.0;
  for (auto score : scores) {
    total_score += score;
  }
  const auto avg_score = total_score / static_cast<double>(scores.size());
  double stddev = 0.0; // Compute the sample standard deviation
  for (auto i=0; i<scores.size(); ++i) {
    stddev += (scores[i] - avg_score) * (scores[i] - avg_score);
  }
  stddev = sqrt(stddev / static_cast<double>(FLAGS_repeat_games - 1));
  LOG(INFO) << "Evaluation avg_score = " << avg_score << " std = " << stddev;
  return avg_score;
}

int main(int argc, char** argv) {
 std::string usage(argv[0]);
  usage.append(" -rom rom -[evaluate|save path]");
  gflags::SetUsageMessage(usage);
  gflags::SetVersionString("0.1");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  if (FLAGS_evaluate) {
    google::LogToStderr();
  }
  if (FLAGS_rom.empty()) {
    LOG(ERROR) << "Rom file required but not set.";
    LOG(ERROR) << "Usage: " << gflags::ProgramUsage();
    exit(1);
  }
  path rom_file(FLAGS_rom);
  if (!is_regular_file(rom_file)) {
    LOG(ERROR) << "Invalid ROM file: " << FLAGS_rom;
    exit(1);
  }
  if (!is_regular_file(FLAGS_solver)) {
    LOG(ERROR) << "Invalid solver: " << FLAGS_solver;
    exit(1);
  }
  if (FLAGS_save.empty() && !FLAGS_evaluate) {
    LOG(ERROR) << "Save path (or evaluate) required but not set.";
    LOG(ERROR) << "Usage: " << gflags::ProgramUsage();
    exit(1);
  }
  path save_path(FLAGS_save);
  if (!FLAGS_evaluate) {
    path snapshot_dir(current_path());
    if (is_directory(save_path)) {
      snapshot_dir = save_path;
      save_path /= rom_file.stem();
    } else {
      if (save_path.has_parent_path()) {
        snapshot_dir = save_path.parent_path();
      }
      save_path += "_";
      save_path += rom_file.stem();
    }
    // Check for files that may be overwritten
    assert(is_directory(snapshot_dir));
    LOG(INFO) << "Snapshots Prefix: " << save_path;
    directory_iterator end;
    for(directory_iterator it(snapshot_dir); it!=end; ++it) {
      if(boost::filesystem::is_regular_file(it->status())) {
        std::string save_path_str = save_path.stem().native();
        std::string other_str = it->path().filename().native();
        auto res = std::mismatch(save_path_str.begin(),
                                 save_path_str.end(),
                                 other_str.begin());
        if (res.first == save_path_str.end()) {
          LOG(ERROR) << "Existing file " << it->path()
                     << " conflicts with save path " << save_path;
          LOG(ERROR) << "Please remove this file or specify another save path.";
          exit(1);
        }
      }
    }
    // Set the logging destinations
    google::SetLogDestination(google::GLOG_INFO,
                              (save_path.native() + "_INFO_").c_str());
    google::SetLogDestination(google::GLOG_WARNING,
                              (save_path.native() + "_WARNING_").c_str());
    google::SetLogDestination(google::GLOG_ERROR,
                              (save_path.native() + "_ERROR_").c_str());
    google::SetLogDestination(google::GLOG_FATAL,
                              (save_path.native() + "_FATAL_").c_str());
  }

  if (FLAGS_gpu) {
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    if (FLAGS_device >= 0) {
      caffe::Caffe::SetDevice(FLAGS_device);
    }
  } else {
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }

  ALEInterface ale;
  InitializeALE(ale, FLAGS_gui, FLAGS_rom);

  // Get the vector of legal actions
  const auto legal_actions = ale.getMinimalActionSet();

  CHECK(FLAGS_snapshot.empty() || FLAGS_weights.empty())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";

  // Construct the solver
  caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(FLAGS_solver, &solver_param);
  caffe::NetParameter* net_param = solver_param.mutable_net_param();
  net_param->CopyFrom(dqn::CreateNet());
  std::string net_filename = save_path.native() + "_net.prototxt";
  WriteProtoToTextFile(*net_param, net_filename.c_str());

  solver_param.set_snapshot_prefix(save_path.c_str());

  dqn::DQN dqn(legal_actions, solver_param, FLAGS_memory, FLAGS_gamma,
               FLAGS_clone_freq);
  dqn.Initialize();

  if (!FLAGS_save_screen.empty()) {
    LOG(INFO) << "Saving screens to: " << FLAGS_save_screen;
  }

  if (!FLAGS_snapshot.empty()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    dqn.RestoreSolver(FLAGS_snapshot);
  } else if (!FLAGS_weights.empty()) {
    LOG(INFO) << "Finetuning from " << FLAGS_weights;
    dqn.LoadTrainedModel(FLAGS_weights);
  }

  if (FLAGS_evaluate) {
    if (FLAGS_gui) {
      auto score = PlayOneEpisode(ale, dqn, FLAGS_evaluate_with_epsilon, false);
      LOG(INFO) << "Score " << score;
    } else {
      Evaluate(ale, dqn);
    }
    return 0;
  }

  dqn.Snapshot();
  int last_eval_iter = 0;
  int episode = 0;
  double best_score = std::numeric_limits<double>::lowest();
  while (dqn.current_iteration() < solver_param.max_iter()) {
    double epsilon = CalculateEpsilon(dqn.current_iteration());
    double score = PlayOneEpisode(ale, dqn, epsilon, true);
    LOG(INFO) << "Episode " << episode << " score = " << score
              << ", epsilon = " << epsilon
              << ", iter = " << dqn.current_iteration()
              << ", replay_mem_size = " << dqn.memory_size();
    episode++;

    // If the size of replay memory is large enough, update DQN
    // if (dqn.memory_size() >= FLAGS_memory_threshold) {
    //   dqn.Update();
    //   LOG(INFO) << "Finished Update iter = " << dqn.current_iteration();
    // }

    if (dqn.current_iteration() >= last_eval_iter + FLAGS_evaluate_freq) {
      double avg_score = Evaluate(ale, dqn);
      if (avg_score > best_score) {
        LOG(INFO) << "iter " << dqn.current_iteration()
                  << " New High Score: " << avg_score;
        best_score = avg_score;
        dqn.Snapshot();
      }
      last_eval_iter = dqn.current_iteration();
    }
  }
  if (dqn.current_iteration() >= last_eval_iter) {
    Evaluate(ale, dqn);
  }
}
