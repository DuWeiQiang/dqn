#include <cmath>
#include <iostream>
#include <ale_interface.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include "prettyprint.hpp"
#include "dqn.hpp"

DEFINE_bool(gpu, true, "Use GPU to brew Caffe");
DEFINE_bool(gui, false, "Open a GUI window");
DEFINE_string(rom, "roms/pong.bin", "Atari 2600 ROM to play");
DEFINE_int32(memory, 500000, "Capacity of replay memory");
DEFINE_int32(explore, 1000000, "Number of iterations needed for epsilon to reach 0.1");
DEFINE_double(gamma, 0.95, "Discount factor of future rewards (0,1]");
DEFINE_int32(memory_threshold, 100, "Enough amount of transitions to start learning");
DEFINE_int32(skip_frame, 3, "Number of frames skipped");
DEFINE_bool(show_frame, false, "Show the current frame in CUI");
DEFINE_string(save_screen, "", "File prefix in to save frames");
DEFINE_string(save_binary_screen, "", "File prefix in to save binary frames");
DEFINE_string(model, "", "Model file to load");
DEFINE_bool(evaluate, false, "Evaluation mode: only playing a game, no updates");
DEFINE_double(evaluate_with_epsilon, 0.05, "Epsilon value to be used in evaluation mode");
DEFINE_double(repeat_games, 1, "Number of games played in evaluation mode");
// Solver Parameters
DEFINE_string(solver, "", "Solver parameter file (*.prototxt)");
DEFINE_string(net, "dqn.prototxt", "Network parameter file (*.prototxt)");
DEFINE_double(momentum, 0.95, "Solver momentum");
DEFINE_double(base_lr, 0.2, "Solver base learning rate");
DEFINE_string(lr_policy, "step", "Solver lr policy");
DEFINE_double(solver_gamma, 0.1, "Solver gamma");
DEFINE_int32(stepsize, 1000000, "Solver stepsize");
DEFINE_int32(max_iter, 2000000, "Maximum number of iterations");
DEFINE_int32(snapshot, 100000, "Snapshot frequency in iterations");
DEFINE_int32(display, 10000, "Display frequency in iterations");
DEFINE_string(snapshot_prefix, "state/dqn", "Prefix for saving snapshots");

double CalculateEpsilon(const int iter) {
  if (iter < FLAGS_explore) {
    return 1.0 - 0.9 * (static_cast<double>(iter) / FLAGS_explore);
  } else {
    return 0.1;
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
double PlayOneEpisode(
    ALEInterface& ale,
    dqn::DQN& dqn,
    const double epsilon,
    const bool update) {
  assert(!ale.game_over());
  std::deque<dqn::FrameDataSp> past_frames;
  auto total_score = 0.0;
  for (auto frame = 0; !ale.game_over(); ++frame) {
    // std::cout << "frame: " << frame << std::endl;
    const ALEScreen& screen = ale.getScreen();
    const auto current_frame = dqn::PreprocessScreen(screen);
    if (FLAGS_show_frame) {
      std::cout << dqn::DrawFrame(*current_frame) << std::endl;
    }
    if (!FLAGS_save_screen.empty()) {
      std::stringstream ss;
      ss << FLAGS_save_screen << setfill('0') << setw(5) <<
          std::to_string(frame) << ".png";
      SaveScreen(screen, ale, ss.str());
    }
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
        // Last action is repeated on skipped frames
        immediate_score += ale.act(action);
      }
      total_score += immediate_score;
      // Rewards for DQN are normalized as follows:
      // 1 for any positive score, -1 for any negative score, otherwise 0
      const auto reward =
          immediate_score == 0 ?
              0 :
              immediate_score /= std::abs(immediate_score);
      if (update) {
        // Add the current transition to replay memory
        const auto transition = ale.game_over() ?
            dqn::Transition(input_frames, action, reward, boost::none) :
            dqn::Transition(
                input_frames,
                action,
                reward,
                dqn::PreprocessScreen(ale.getScreen()));
        dqn.AddTransition(transition);
        // If the size of replay memory is enough, update DQN
        if (dqn.memory_size() > FLAGS_memory_threshold) {
          dqn.Update();
        }
      }
    }
  }
  ale.reset_game();
  return total_score;
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  google::LogToStderr();

  if (FLAGS_gpu) {
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
  } else {
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }

  ALEInterface ale;
  ale.set("display_screen", FLAGS_gui);
  ale.set("disable_color_averaging", "true");

  // Load the ROM file
  ale.loadROM(FLAGS_rom);

  // Get the vector of legal actions
  const auto legal_actions = ale.getMinimalActionSet();

  // Construct the solver either from file or params
  caffe::SolverParameter solver_param;
  if (!FLAGS_solver.empty()) {
    std::cout << "Loading solver from " << FLAGS_solver << std::endl;
    caffe::ReadProtoFromTextFileOrDie(FLAGS_solver, &solver_param);
  } else {
    solver_param.set_net(FLAGS_net);
    solver_param.set_solver_type(::caffe::SolverParameter_SolverType::
                                 SolverParameter_SolverType_ADADELTA);
    solver_param.set_momentum(FLAGS_momentum);
    solver_param.set_base_lr(FLAGS_base_lr);
    solver_param.set_lr_policy(FLAGS_lr_policy);
    solver_param.set_gamma(FLAGS_solver_gamma);
    solver_param.set_stepsize(FLAGS_stepsize);
    solver_param.set_max_iter(FLAGS_max_iter);
    solver_param.set_display(FLAGS_display);
    solver_param.set_snapshot(FLAGS_snapshot);
    solver_param.set_snapshot_prefix(FLAGS_snapshot_prefix);
  }

  dqn::DQN dqn(legal_actions, solver_param, FLAGS_memory, FLAGS_gamma);
  dqn.Initialize();

  if (!FLAGS_save_screen.empty()) {
    std::cout << "Saving screens to: " << FLAGS_save_screen << std::endl;
  }

  if (!FLAGS_model.empty()) {
    // Just evaluate the given trained model
    std::cout << "Loading " << FLAGS_model << std::endl;
    dqn.LoadTrainedModel(FLAGS_model);
  }

  if (FLAGS_evaluate) {
    auto total_score = 0.0;
    for (auto i = 0; i < FLAGS_repeat_games; ++i) {
      std::cout << "game: " << i << std::endl;
      const auto score =
          PlayOneEpisode(ale, dqn, FLAGS_evaluate_with_epsilon, false);
      std::cout << "score: " << score << std::endl;
      total_score += score;
    }
    std::cout << "total_score: " << total_score << std::endl;
    return 0;
  }

  for (auto episode = 0; dqn.current_iteration() < FLAGS_max_iter; episode++) {
    const auto epsilon = CalculateEpsilon(dqn.current_iteration());
    std::cout << "Episode " << episode
              << ", epsilon = " << epsilon << std::endl;
    PlayOneEpisode(ale, dqn, epsilon, true);
    if (dqn.current_iteration() % 10 == 0) {
      // After every 10 episodes, evaluate the current strength
      const auto eval_score = PlayOneEpisode(ale, dqn, 0.05, false);
      std::cout << "evaluation score: " << eval_score << std::endl;
    }
  }
};
