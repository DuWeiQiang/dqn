#include <cmath>
#include <iostream>
#include <ale_interface.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include "prettyprint.hpp"
#include "dqn.hpp"
#include <boost/filesystem.hpp>

using namespace boost::filesystem;

// DQN Parameters
DEFINE_bool(gpu, true, "Use GPU to brew Caffe");
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
DEFINE_int32(repeat_games, 10, "Number of games played in evaluation mode");
DEFINE_string(solver, "dqn_solver.prototxt", "Solver parameter file (*.prototxt)");

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

void SaveInputFrames(const dqn::InputFrames& frames, const string filename) {
  std::ofstream ofs;
  ofs.open(filename, ios::out | ios::binary);
  for (int i = 0; i < dqn::kInputFrameCount; ++i) {
    const dqn::FrameData& frame = *frames[i];
    for (int j = 0; j < dqn::kCroppedFrameDataSize; ++j) {
      ofs.write((char*) &frame[j], sizeof(uint8_t));
    }
  }
  ofs.close();
}

/**
 * Play one episode and return the total score
 */
double PlayOneEpisode(ALEInterface& ale, dqn::DQN& dqn, const double epsilon,
                      const bool update) {
  CHECK(!ale.game_over());
  std::deque<dqn::FrameDataSp> past_frames;
  auto total_score = 0.0;
  for (auto frame = 0; !ale.game_over(); ++frame) {
    const ALEScreen& screen = ale.getScreen();
    if (!FLAGS_save_screen.empty()) {
      std::stringstream ss;
      ss << FLAGS_save_screen << setfill('0') << setw(5) <<
          std::to_string(frame) << ".png";
      SaveScreen(screen, ale, ss.str());
    }
    const auto current_frame = dqn::PreprocessScreen(screen);
    past_frames.push_back(current_frame);
    if (past_frames.size() < dqn::kInputFrameCount) {
      // If there are not past frames enough for DQN input, just select NOOP
      for (auto i = 0; i < FLAGS_skip_frame + 1 && !ale.game_over(); ++i) {
        total_score += ale.act(PLAYER_A_NOOP);
      }
    } else {
      while (past_frames.size() > dqn::kInputFrameCount) {
        past_frames.pop_front();
      }
      dqn::InputFrames input_frames;
      std::copy(past_frames.begin(), past_frames.end(), input_frames.begin());
      if (!FLAGS_save_binary_screen.empty()) {
        static int binary_save_num = 0;
        string fname = FLAGS_save_binary_screen +
            std::to_string(binary_save_num++) + ".bin";
        SaveInputFrames(input_frames, fname);
      }
      const auto action = dqn.SelectAction(input_frames, epsilon);
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
            dqn::Transition(input_frames, action, reward, boost::none) :
            dqn::Transition(input_frames, action, reward,
                            dqn::PreprocessScreen(ale.getScreen()));
        dqn.AddTransition(transition);
        // If the size of replay memory is large enough, update DQN
        if (dqn.memory_size() > FLAGS_memory_threshold) {
          dqn.Update();
        }
      }
    }
  }
  ale.reset_game();
  return total_score;
}

/**
 * Evaluate the current player
 */
void Evaluate(ALEInterface& ale, dqn::DQN& dqn) {
  auto total_score = 0.0;
  std::stringstream ss;
  for (auto i = 0; i < FLAGS_repeat_games; ++i) {
    const auto score =
        PlayOneEpisode(ale, dqn, FLAGS_evaluate_with_epsilon, false);
    ss << score << " ";
    total_score += score;
  }
  LOG(INFO) << "Evaluation scores: " << ss.str();
  LOG(INFO) << "Average score: " <<
      total_score / static_cast<double>(FLAGS_repeat_games) << std::endl;
}

int main(int argc, char** argv) {
  std::string usage(argv[0]);
  usage.append(" -rom rom -[evaluate|save path]");
  gflags::SetUsageMessage(usage);
  gflags::SetVersionString("0.1");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  google::LogToStderr();

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

  if (FLAGS_gpu) {
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
  } else {
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }

  ALEInterface ale;
  ale.set("display_screen", FLAGS_gui);
  ale.set("disable_color_averaging", true);

  // Load the ROM file
  ale.loadROM(FLAGS_rom);

  // Get the vector of legal actions
  const auto legal_actions = ale.getMinimalActionSet();

  CHECK(FLAGS_snapshot.empty() || FLAGS_weights.empty())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";

  // Construct the solver
  caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(FLAGS_solver, &solver_param);
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
    Evaluate(ale, dqn);
    return 0;
  }

  auto episode = 0;
  while (dqn.current_iteration() < solver_param.max_iter()) {
    const auto epsilon = CalculateEpsilon(dqn.current_iteration());
    const auto score = PlayOneEpisode(ale, dqn, epsilon, true);
    LOG(INFO) << "Episode " << episode << ", score = " << score
              << ", epsilon = " << epsilon << ", iter = "
              << dqn.current_iteration();
    episode++;
  }
  Evaluate(ale, dqn);
};
