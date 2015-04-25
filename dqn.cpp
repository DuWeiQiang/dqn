#include "dqn.hpp"
#include <algorithm>
#include <iostream>
#include <cassert>
#include <sstream>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <glog/logging.h>
#include "prettyprint.hpp"

namespace dqn {

void SaveInputFrame(const dqn::FrameData& frame, const string filename) {
  std::ofstream ofs;
  ofs.open(filename, ios::out | ios::binary);
  for (int i = 0; i < dqn::kCroppedFrameDataSize; ++i) {
    ofs.write((char*) &frame[i], sizeof(uint8_t));
  }
  ofs.close();
}

/**
 * Convert pixel_t (NTSC) to RGB values.
 * Each value range [0,255]
 */
const std::array<int, 3> PixelToRGB(const pixel_t& pixel) {
  constexpr int ntsc_to_rgb[] = {
    0x000000, 0, 0x4a4a4a, 0, 0x6f6f6f, 0, 0x8e8e8e, 0,
    0xaaaaaa, 0, 0xc0c0c0, 0, 0xd6d6d6, 0, 0xececec, 0,
    0x484800, 0, 0x69690f, 0, 0x86861d, 0, 0xa2a22a, 0,
    0xbbbb35, 0, 0xd2d240, 0, 0xe8e84a, 0, 0xfcfc54, 0,
    0x7c2c00, 0, 0x904811, 0, 0xa26221, 0, 0xb47a30, 0,
    0xc3903d, 0, 0xd2a44a, 0, 0xdfb755, 0, 0xecc860, 0,
    0x901c00, 0, 0xa33915, 0, 0xb55328, 0, 0xc66c3a, 0,
    0xd5824a, 0, 0xe39759, 0, 0xf0aa67, 0, 0xfcbc74, 0,
    0x940000, 0, 0xa71a1a, 0, 0xb83232, 0, 0xc84848, 0,
    0xd65c5c, 0, 0xe46f6f, 0, 0xf08080, 0, 0xfc9090, 0,
    0x840064, 0, 0x97197a, 0, 0xa8308f, 0, 0xb846a2, 0,
    0xc659b3, 0, 0xd46cc3, 0, 0xe07cd2, 0, 0xec8ce0, 0,
    0x500084, 0, 0x68199a, 0, 0x7d30ad, 0, 0x9246c0, 0,
    0xa459d0, 0, 0xb56ce0, 0, 0xc57cee, 0, 0xd48cfc, 0,
    0x140090, 0, 0x331aa3, 0, 0x4e32b5, 0, 0x6848c6, 0,
    0x7f5cd5, 0, 0x956fe3, 0, 0xa980f0, 0, 0xbc90fc, 0,
    0x000094, 0, 0x181aa7, 0, 0x2d32b8, 0, 0x4248c8, 0,
    0x545cd6, 0, 0x656fe4, 0, 0x7580f0, 0, 0x8490fc, 0,
    0x001c88, 0, 0x183b9d, 0, 0x2d57b0, 0, 0x4272c2, 0,
    0x548ad2, 0, 0x65a0e1, 0, 0x75b5ef, 0, 0x84c8fc, 0,
    0x003064, 0, 0x185080, 0, 0x2d6d98, 0, 0x4288b0, 0,
    0x54a0c5, 0, 0x65b7d9, 0, 0x75cceb, 0, 0x84e0fc, 0,
    0x004030, 0, 0x18624e, 0, 0x2d8169, 0, 0x429e82, 0,
    0x54b899, 0, 0x65d1ae, 0, 0x75e7c2, 0, 0x84fcd4, 0,
    0x004400, 0, 0x1a661a, 0, 0x328432, 0, 0x48a048, 0,
    0x5cba5c, 0, 0x6fd26f, 0, 0x80e880, 0, 0x90fc90, 0,
    0x143c00, 0, 0x355f18, 0, 0x527e2d, 0, 0x6e9c42, 0,
    0x87b754, 0, 0x9ed065, 0, 0xb4e775, 0, 0xc8fc84, 0,
    0x303800, 0, 0x505916, 0, 0x6d762b, 0, 0x88923e, 0,
    0xa0ab4f, 0, 0xb7c25f, 0, 0xccd86e, 0, 0xe0ec7c, 0,
    0x482c00, 0, 0x694d14, 0, 0x866a26, 0, 0xa28638, 0,
    0xbb9f47, 0, 0xd2b656, 0, 0xe8cc63, 0, 0xfce070, 0
  };
  const auto rgb = ntsc_to_rgb[pixel];
  const auto r = rgb >> 16;
  const auto g = (rgb >> 8) & 0xFF;
  const auto b = rgb & 0xFF;
  std::array<int, 3> arr = {r, g, b};
  return arr;
}

/**
 * Convert RGB values to a grayscale value [0,255].
 */
uint8_t RGBToGrayscale(const std::array<int, 3>& rgb) {
  CHECK(rgb[0] >= 0 && rgb[0] <= 255);
  CHECK(rgb[1] >= 0 && rgb[1] <= 255);
  CHECK(rgb[2] >= 0 && rgb[2] <= 255);
  // Normalized luminosity grayscale
  return rgb[0] * 0.21 + rgb[1] * 0.72 + rgb[2] * 0.07;
}

uint8_t PixelToGrayscale(const pixel_t pixel) {
  return RGBToGrayscale(PixelToRGB(pixel));
}

FrameDataSp PreprocessScreen(const ALEScreen& raw_screen) {
  const int raw_screen_width = raw_screen.width();
  const int raw_screen_height = raw_screen.height();
  CHECK_GT(raw_screen_height, raw_screen_width);
  const auto raw_pixels = raw_screen.getArray();
  auto screen = std::make_shared<FrameData>();
  // Crop the top of the screen
  const int cropped_screen_height = static_cast<int>(.85 * raw_screen_height);
  const int start_y = raw_screen_height - cropped_screen_height;
  // Ignore the leftmost column of 8 pixels
  const int start_x = 8;
  const int cropped_screen_width = raw_screen_width - start_x;
  const auto x_ratio = cropped_screen_width / static_cast<double>(kCroppedFrameSize);
  const auto y_ratio = cropped_screen_height / static_cast<double>(kCroppedFrameSize);
  for (auto i = 0; i < kCroppedFrameSize; ++i) {
    for (auto j = 0; j < kCroppedFrameSize; ++j) {
      const auto first_x = start_x + static_cast<int>(std::floor(j * x_ratio));
      const auto last_x = start_x + static_cast<int>(std::floor((j + 1) * x_ratio));
      const auto first_y = start_y + static_cast<int>(std::floor(i * y_ratio));
      const auto last_y = start_y + static_cast<int>(std::floor((i + 1) * y_ratio));
      uint8_t resulting_color = 0.0;
      for (auto x = first_x; x <= last_x; ++x) {
        double x_ratio_in_resulting_pixel = 1.0;
        if (x == first_x) {
          x_ratio_in_resulting_pixel = x + 1 - j * x_ratio - start_x;
        } else if (x == last_x) {
          x_ratio_in_resulting_pixel = x_ratio * (j + 1) - x + start_x;
        }
        assert(x_ratio_in_resulting_pixel >= 0.0 &&
               x_ratio_in_resulting_pixel <= 1.0);
        for (auto y = first_y; y <= last_y; ++y) {
          double y_ratio_in_resulting_pixel = 1.0;
          if (y == first_y) {
            y_ratio_in_resulting_pixel = y + 1 - i * y_ratio - start_y;
          } else if (y == last_y) {
            y_ratio_in_resulting_pixel = y_ratio * (i + 1) - y + start_y;
          }
          assert(y_ratio_in_resulting_pixel >= 0.0 &&
                 y_ratio_in_resulting_pixel <= 1.0);
          const auto grayscale =
              PixelToGrayscale(
                  raw_pixels[static_cast<int>(y * raw_screen_width + x)]);
          resulting_color +=
              (x_ratio_in_resulting_pixel / x_ratio) *
              (y_ratio_in_resulting_pixel / y_ratio) * grayscale;
        }
      }
      (*screen)[i * kCroppedFrameSize + j] = resulting_color;
    }
  }
  return screen;
}

void PrintQValues(const std::vector<float>& q_values, const ActionVect& actions) {
  CHECK_GT(q_values.size(), 0);
  CHECK_GT(actions.size(), 0);
  CHECK_EQ(q_values.size(), actions.size());
  std::ostringstream actions_buf;
  std::ostringstream q_values_buf;
  for (auto i = 0; i < q_values.size(); ++i) {
    const auto a_str =
        boost::algorithm::replace_all_copy(
            action_to_string(actions[i]), "PLAYER_A_", "");
    const auto q_str = std::to_string(q_values[i]);
    const auto column_size = std::max(a_str.size(), q_str.size()) + 1;
    actions_buf.width(column_size);
    actions_buf << a_str;
    q_values_buf.width(column_size);
    q_values_buf << q_str;
  }
  LOG(INFO) << actions_buf.str();
  LOG(INFO) << q_values_buf.str();
}

template <typename Dtype>
void HasBlobSize(caffe::Net<Dtype>& net,
                 const std::string& blob_name,
                 const std::vector<int> expected_shape) {
  net.has_blob(blob_name);
  const caffe::Blob<Dtype>& blob = *net.blob_by_name(blob_name);
  const std::vector<int>& blob_shape = blob.shape();
  CHECK_EQ(blob_shape.size(), expected_shape.size());
  CHECK(std::equal(blob_shape.begin(), blob_shape.end(),
                   expected_shape.begin()));
}

void PopulateLayer(caffe::LayerParameter& layer,
                   const std::string& name, const std::string& type,
                   const std::vector<std::string>& bottoms,
                   const std::vector<std::string>& tops,
                   const boost::optional<caffe::Phase>& include_phase) {
  layer.set_name(name);
  layer.set_type(type);
  for (auto& bottom : bottoms) {
    layer.add_bottom(bottom);
  }
  for (auto& top : tops) {
    layer.add_top(top);
  }
  // PopulateLayer(layer, name, type, bottoms, tops);
  if (include_phase) {
    layer.add_include()->set_phase(*include_phase);
  }
}

void MemoryDataLayer(caffe::NetParameter& net_param,
                     const std::string& name,
                     const std::vector<std::string>& tops,
                     const boost::optional<caffe::Phase>& include_phase,
                     const std::vector<int>& shape) {
  caffe::LayerParameter& memory_layer = *net_param.add_layer();
  PopulateLayer(memory_layer, name, "MemoryData", {}, tops, include_phase);
  CHECK_EQ(shape.size(), 4);
  caffe::MemoryDataParameter* memory_data_param =
      memory_layer.mutable_memory_data_param();
  memory_data_param->set_batch_size(shape[0]);
  memory_data_param->set_channels(shape[1]);
  memory_data_param->set_height(shape[2]);
  memory_data_param->set_width(shape[3]);
}

void ReshapeLayer(caffe::NetParameter& net_param,
                  const std::string& name,
                  const std::vector<std::string>& bottoms,
                  const std::vector<std::string>& tops,
                  const boost::optional<caffe::Phase>& include_phase,
                  const std::vector<int>& shape) {
  caffe::LayerParameter& reshape_layer = *net_param.add_layer();
  PopulateLayer(reshape_layer, name, "Reshape", bottoms, tops, include_phase);
  caffe::ReshapeParameter* reshape_param = reshape_layer.mutable_reshape_param();
  caffe::BlobShape* blob_shape = reshape_param->mutable_shape();
  for (auto& dim : shape) {
    blob_shape->add_dim(dim);
  }
}

void SliceLayer(caffe::NetParameter& net_param,
                const std::string& name,
                const std::vector<std::string>& bottoms,
                const std::vector<std::string>& tops,
                const boost::optional<caffe::Phase>& include_phase,
                const int axis,
                const std::vector<int>& slice_points) {
  caffe::LayerParameter& layer = *net_param.add_layer();
  PopulateLayer(layer, name, "Slice", bottoms, tops, include_phase);
  caffe::SliceParameter* slice_param = layer.mutable_slice_param();
  slice_param->set_axis(axis);
  for (auto& p : slice_points) {
    slice_param->add_slice_point(p);
  }
}

void ConvLayer(caffe::NetParameter& net_param,
               const std::string& name,
               const std::vector<std::string>& bottoms,
               const std::vector<std::string>& tops,
               const std::string& shared_name,
               const boost::optional<caffe::Phase>& include_phase,
               const int num_output,
               const int kernel_size,
               const int stride) {
  caffe::LayerParameter& layer = *net_param.add_layer();
  PopulateLayer(layer, name, "Convolution", bottoms, tops, include_phase);
  caffe::ParamSpec* weight_param = layer.add_param();
  weight_param->set_name(shared_name + "_w");
  weight_param->set_decay_mult(1);
  caffe::ParamSpec* bias_param = layer.add_param();
  bias_param->set_name(shared_name + "_b");
  bias_param->set_decay_mult(0);
  caffe::ConvolutionParameter* conv_param = layer.mutable_convolution_param();
  conv_param->set_num_output(num_output);
  conv_param->set_kernel_size(kernel_size);
  conv_param->set_stride(stride);
  caffe::FillerParameter* weight_filler = conv_param->mutable_weight_filler();
  weight_filler->set_type("gaussian");
  weight_filler->set_std(0.01);
  caffe::FillerParameter* bias_filler = conv_param->mutable_bias_filler();
  bias_filler->set_type("constant");
  bias_filler->set_value(0);
}

void ReluLayer(caffe::NetParameter& net_param,
               const std::string& name,
               const std::vector<std::string>& bottoms,
               const std::vector<std::string>& tops,
               const boost::optional<caffe::Phase>& include_phase) {
  caffe::LayerParameter& layer = *net_param.add_layer();
  PopulateLayer(layer, name, "ReLU", bottoms, tops, include_phase);
  caffe::ReLUParameter* relu_param = layer.mutable_relu_param();
  relu_param->set_negative_slope(0.01);
}

void IPLayer(caffe::NetParameter& net_param,
             const std::string& name,
             const std::vector<std::string>& bottoms,
             const std::vector<std::string>& tops,
             const std::string& shared_name,
             const boost::optional<caffe::Phase>& include_phase,
             const int num_output,
             const int axis) {
  caffe::LayerParameter& layer = *net_param.add_layer();
  PopulateLayer(layer, name, "InnerProduct", bottoms, tops, include_phase);
  caffe::ParamSpec* weight_param = layer.add_param();
  weight_param->set_name(shared_name + "_w");
  weight_param->set_decay_mult(1);
  caffe::ParamSpec* bias_param = layer.add_param();
  bias_param->set_name(shared_name + "_b");
  bias_param->set_decay_mult(0);
  caffe::InnerProductParameter* ip_param = layer.mutable_inner_product_param();
  ip_param->set_num_output(num_output);
  ip_param->set_axis(axis);
  caffe::FillerParameter* weight_filler = ip_param->mutable_weight_filler();
  weight_filler->set_type("gaussian");
  weight_filler->set_std(0.005);
  caffe::FillerParameter* bias_filler = ip_param->mutable_bias_filler();
  bias_filler->set_type("constant");
  bias_filler->set_value(1);
}

void ConcatLayer(caffe::NetParameter& net_param,
                 const std::string& name,
                 const std::vector<std::string>& bottoms,
                 const std::vector<std::string>& tops,
                 const boost::optional<caffe::Phase>& include_phase,
                 const int& axis) {
  caffe::LayerParameter& layer = *net_param.add_layer();
  PopulateLayer(layer, name, "Concat", bottoms, tops, include_phase);
  caffe::ConcatParameter* concat_param = layer.mutable_concat_param();
  concat_param->set_axis(axis);
}

void LstmLayer(caffe::NetParameter& net_param,
               const std::string& name,
               const std::vector<std::string>& bottoms,
               const std::vector<std::string>& tops,
               const boost::optional<caffe::Phase>& include_phase,
               const int& num_output) {
  caffe::LayerParameter& layer = *net_param.add_layer();
  PopulateLayer(layer, name, "LSTM", bottoms, tops, include_phase);
  caffe::RecurrentParameter* recurrent_param = layer.mutable_recurrent_param();
  recurrent_param->set_num_output(num_output);
  caffe::FillerParameter* weight_filler = recurrent_param->mutable_weight_filler();
  weight_filler->set_type("uniform");
  weight_filler->set_min(-0.08);
  weight_filler->set_max(0.08);
  caffe::FillerParameter* bias_filler = recurrent_param->mutable_bias_filler();
  bias_filler->set_type("constant");
  bias_filler->set_value(0);
}

void EltwiseLayer(caffe::NetParameter& net_param,
                  const std::string& name,
                  const std::vector<std::string>& bottoms,
                  const std::vector<std::string>& tops,
                  const boost::optional<caffe::Phase>& include_phase,
                  const caffe::EltwiseParameter::EltwiseOp& op) {
  caffe::LayerParameter& layer = *net_param.add_layer();
  PopulateLayer(layer, name, "Eltwise", bottoms, tops, include_phase);
  caffe::EltwiseParameter* eltwise_param = layer.mutable_eltwise_param();
  eltwise_param->set_operation(op);
}

void SilenceLayer(caffe::NetParameter& net_param,
                  const std::string& name,
                  const std::vector<std::string>& bottoms,
                  const std::vector<std::string>& tops,
                  const boost::optional<caffe::Phase>& include_phase) {
  caffe::LayerParameter& layer = *net_param.add_layer();
  PopulateLayer(layer, name, "Silence", bottoms, tops, include_phase);
}

void EuclideanLossLayer(caffe::NetParameter& net_param,
                        const std::string& name,
                        const std::vector<std::string>& bottoms,
                        const std::vector<std::string>& tops,
                        const boost::optional<caffe::Phase>& include_phase) {
  caffe::LayerParameter& layer = *net_param.add_layer();
  PopulateLayer(layer, name, "EuclideanLoss", bottoms, tops, include_phase);
}

caffe::NetParameter CreateNet() {
  caffe::NetParameter np;
  np.set_name("Deep Recurrent Q-Network");
  MemoryDataLayer(
      np, frames_layer_name, {train_frames_blob_name,"dummy_frames"}, caffe::TRAIN,
      {kMinibatchSize, kMemoryInputFrames, kCroppedFrameSize, kCroppedFrameSize});
  MemoryDataLayer(
      np, cont_layer_name, {cont_blob_name,"dummy_cont"}, caffe::TRAIN,
      {kUnroll, kMinibatchSize, 1, 1});
  MemoryDataLayer(
      np, target_layer_name, {target_blob_name,"dummy_target"}, caffe::TRAIN,
      {kUnroll, kMinibatchSize, kOutputCount, 1});
  MemoryDataLayer(
      np, filter_layer_name, {filter_blob_name,"dummy_filter"}, caffe::TRAIN,
      {kUnroll, kMinibatchSize, kOutputCount, 1});
  ReshapeLayer(
      np, "reshape_cont", {cont_blob_name}, {"reshaped_cont"}, caffe::TRAIN,
      {kUnroll, kMinibatchSize});
  ReshapeLayer(
      np, "reshape_filter", {filter_blob_name}, {"reshaped_filter"}, caffe::TRAIN,
      {kUnroll, kMinibatchSize, kOutputCount});
  MemoryDataLayer(
      np, frames_layer_name, {test_frames_blob_name,"dummy_frames"},
      caffe::TEST,
      {kMinibatchSize,kInputFramesPerTimestep,kCroppedFrameSize,kCroppedFrameSize});
  MemoryDataLayer(
      np, cont_layer_name, {cont_blob_name,"dummy_cont"}, caffe::TEST,
      {1, kMinibatchSize, 1, 1});
  MemoryDataLayer(
      np, target_layer_name, {target_blob_name,"dummy_target"}, caffe::TEST,
      {1, kMinibatchSize, kOutputCount, 1});
  MemoryDataLayer(
      np, filter_layer_name, {filter_blob_name,"dummy_filter"}, caffe::TEST,
      {1, kMinibatchSize, kOutputCount, 1});
  ReshapeLayer(
      np, "reshape_cont", {cont_blob_name}, {"reshaped_cont"}, caffe::TEST,
      {1, kMinibatchSize});
  ReshapeLayer(
      np, "reshape_filter", {filter_blob_name}, {"reshaped_filter"}, caffe::TEST,
      {1, kMinibatchSize, kOutputCount});
  SilenceLayer(np, "silence", {"dummy_frames","dummy_cont","dummy_filter",
          "dummy_target"}, {}, boost::none);
  std::vector<std::string> ip1_tops, scrap_tops;
  for (int t = 0; t < kUnroll; ++t) {
    std::string ts = std::to_string(t);
    boost::optional<caffe::Phase> phase;
    if (t > 0) { phase.reset(caffe::TRAIN); }
    std::vector<int> slice_points;
    std::vector<std::string> slice_tops;
    if (t == 0) {
      slice_points = {kInputFramesPerTimestep};
      slice_tops = {"frames_"+ts, "scrap_"+ts};
      scrap_tops.push_back("scrap_"+ts);
    } else if (t == kUnroll - 1) {
      slice_points = {t};
      slice_tops = {"scrap_"+ts, "frames_"+ts};
      scrap_tops.push_back("scrap_"+ts);
    } else {
      slice_tops = {"scrap1_"+ts, "frames_"+ts, "scrap2_"+ts};
      scrap_tops.push_back("scrap1_"+ts);
      scrap_tops.push_back("scrap2_"+ts);
      slice_points = {t, t + kInputFramesPerTimestep};
    }
    SliceLayer(np, "slice_"+ts, {train_frames_blob_name}, slice_tops,
               caffe::TRAIN, 1, slice_points);
    ConvLayer(np, "conv1_"+ts, {"frames_"+ts}, {"conv1_"+ts}, "conv1", phase,
              16, 8, 4);
    ReluLayer(np, "conv1_relu_"+ts, {"conv1_"+ts}, {"conv1_"+ts}, phase);
    ConvLayer(np, "conv2_"+ts, {"conv1_"+ts}, {"conv2_"+ts}, "conv2", phase,
              32, 4, 2);
    ReluLayer(np, "conv2_relu_"+ts, {"conv2_"+ts}, {"conv2_"+ts}, phase);
    IPLayer(np, "ip1_"+ts, {"conv2_"+ts}, {"ip1_"+ts}, "ip1", phase, ip1Size,1);
    ReluLayer(np, "ip1_relu_"+ts, {"ip1_"+ts}, {"ip1_"+ts}, phase);
    ReshapeLayer(np, "ip1_reshape_"+ts, {"ip1_"+ts}, {"reshaped_ip1_"+ts},
                 caffe::TRAIN, {1, kMinibatchSize, ip1Size});
    ip1_tops.push_back("reshaped_ip1_"+ts);
  }
  SilenceLayer(np, "scrap_silence", scrap_tops, {}, caffe::TRAIN);
  ReshapeLayer(np, "ip1_reshape_0", {"ip1_0"}, {"concat_ip1"}, caffe::TEST,
               {1, kMinibatchSize, ip1Size});
  ConcatLayer(np, "concat_ip1", ip1_tops, {"concat_ip1"}, caffe::TRAIN, 0);
  LstmLayer(np,"lstm1",{"concat_ip1","reshaped_cont"},{"lstm1"},boost::none,lstmSize);
  IPLayer(np,"ip2",{"lstm1"},{q_values_blob_name},"ip2",boost::none,kOutputCount,2);
  EltwiseLayer(np, "eltwise_filter", {q_values_blob_name,"reshaped_filter"},
               {"filtered_q_values"}, boost::none, caffe::EltwiseParameter::PROD);
  EuclideanLossLayer(np, "loss", {"filtered_q_values","target"}, {"loss"}, boost::none);
  // WriteProtoToTextFile(np, "gen_net.prototxt");
  // exit(0);
  return np;
}

void DQN::LoadTrainedModel(const std::string& model_bin) {
  net_->CopyTrainedLayersFrom(model_bin);
}

void DQN::RestoreSolver(const std::string& solver_bin) {
  solver_->Restore(solver_bin.c_str());
}

void DQN::Initialize() {
  // Initialize net and solver
  solver_.reset(caffe::GetSolver<float>(solver_param_));
  net_ = solver_->net();
  CHECK_EQ(solver_->test_nets().size(), 1);
  test_net_ = solver_->test_nets()[0];
  // Test Net shares parameters with train net at all times
  test_net_->ShareTrainedLayersWith(net_.get());
  // Clone net maintains its own set of parameters
  CloneNet(*test_net_);
  // Check the primary network
  HasBlobSize(*net_, train_frames_blob_name, {kMinibatchSize,
          kMemoryInputFrames, kCroppedFrameSize, kCroppedFrameSize});
  HasBlobSize(*net_, target_blob_name, {kUnroll, kMinibatchSize, kOutputCount, 1});
  HasBlobSize(*net_, filter_blob_name, {kUnroll, kMinibatchSize, kOutputCount, 1});
  HasBlobSize(*net_, cont_blob_name, {kUnroll, kMinibatchSize, 1, 1});
  // Check the test network
  HasBlobSize(*test_net_, test_frames_blob_name, {kMinibatchSize,
          kInputFramesPerTimestep, kCroppedFrameSize, kCroppedFrameSize});
  HasBlobSize(*test_net_, target_blob_name, {1, kMinibatchSize, kOutputCount, 1});
  HasBlobSize(*test_net_, filter_blob_name, {1, kMinibatchSize, kOutputCount, 1});
  HasBlobSize(*test_net_, cont_blob_name, {1, kMinibatchSize, 1, 1});
  LOG(INFO) << "Finished " << net_->name() << " Initialization";
}

Action DQN::SelectAction(const InputFrames& frames, const double epsilon,
                         bool cont) {
  return SelectActions(std::vector<InputFrames>{{frames}}, epsilon, cont)[0];
}

ActionVect DQN::SelectActions(const std::vector<InputFrames>& frames_batch,
                              const double epsilon, bool cont) {
  CHECK(epsilon <= 1.0 && epsilon >= 0.0);
  CHECK_LE(frames_batch.size(), kMinibatchSize);
  ActionVect actions(frames_batch.size());
  if (std::uniform_real_distribution<>(0.0, 1.0)(random_engine) < epsilon) {
    // Select randomly
    for (int i = 0; i < actions.size(); ++i) {
      const auto random_idx = std::uniform_int_distribution<int>
          (0, legal_actions_.size() - 1)(random_engine);
      actions[i] = legal_actions_[random_idx];
    }
  } else {
    // Select greedily
    std::vector<ActionValue> actions_and_values =
        SelectActionGreedily(*test_net_, frames_batch, cont);
    CHECK_EQ(actions_and_values.size(), actions.size());
    for (int i=0; i<actions_and_values.size(); ++i) {
      actions[i] = actions_and_values[i].first;
    }
  }
  return actions;
}

ActionValue DQN::SelectActionGreedily(caffe::Net<float>& net,
                                      const InputFrames& last_frames,
                                      bool cont) {
  return SelectActionGreedily(
      net, std::vector<InputFrames>{{last_frames}}, cont).front();
}

std::vector<ActionValue>
DQN::SelectActionGreedily(caffe::Net<float>& net,
                          const std::vector<InputFrames>& frames_batch,
                          bool cont) {
  std::vector<ActionValue> results;
  if (frames_batch.empty()) {
    return results;
  }
  CHECK_EQ(net.phase(), caffe::TEST);
  CHECK(net.has_blob(test_frames_blob_name));
  const auto frames_blob = net.blob_by_name(test_frames_blob_name);
  CHECK_LE(frames_batch.size(), kMinibatchSize);
  std::array<float, kTestFramesInputSize> frames_input;
  frames_input.fill(0.0);
  std::array<float, kTestContInputSize> cont_input;
  cont_input.fill(cont);
  FilterLayerInputData filter_input;
  filter_input.fill(0.0);
  TargetLayerInputData target_input;
  target_input.fill(0.0);
  // Input frames to the net and compute Q values for each legal action
  for (int n = 0; n < frames_batch.size(); ++n) {
    const InputFrames& input_frames = frames_batch[n];
    for (int i = 0; i < kInputFramesPerTimestep; ++i) {
      const FrameDataSp& frame_data = input_frames[i];
      std::copy(frame_data->begin(), frame_data->end(),
                frames_input.begin() + frames_blob->offset(n,i,0,0));
    }
  }
  InputDataIntoLayers(net, frames_input.data(), cont_input.data(),
                      target_input.data(), filter_input.data());
  net.ForwardPrefilled(nullptr);
  // Collect the Results
  results.reserve(frames_batch.size());
  CHECK(net.has_blob(q_values_blob_name));
  const auto q_values_blob = net.blob_by_name(q_values_blob_name);
  for (int i = 0; i < frames_batch.size(); ++i) {
    // Get the Q-values from the net: kUnroll*kMinibatchSize*kOutputCount
    const auto action_evaluator = [&](Action action) {
      const auto q = q_values_blob->data_at(0, i, static_cast<int>(action), 0);
      CHECK(!std::isnan(q));
      return q;
    };
    std::vector<float> q_values(legal_actions_.size());
    std::transform(legal_actions_.begin(), legal_actions_.end(),
                   q_values.begin(), action_evaluator);
    // PrintQValues(q_values, legal_actions_);
    // Select the action with the maximum Q value
    const auto max_idx = std::distance(
        q_values.begin(),
        std::max_element(q_values.begin(), q_values.end()));
    results.emplace_back(legal_actions_[max_idx], q_values[max_idx]);
  }
  return results;
}

void DQN::RememberEpisode(const Episode& episode) {
  replay_memory_size_ += episode.size();
  replay_memory_.push_back(episode);
  while (replay_memory_size_ >= replay_memory_capacity_) {
    replay_memory_size_ -= replay_memory_.front().size();
    replay_memory_.pop_front();
  }
}

int DQN::Update() {
  // Every clone_iters steps, update the clone_net_
  if (!clone_net_ || current_iteration() >= last_clone_iter_ + clone_frequency_) {
    LOG(INFO) << "Iter " << current_iteration() << ": Updating Clone Net";
    CloneNet(*test_net_);
    last_clone_iter_ = current_iteration();
  }

  const auto frames_blob = net_->blob_by_name(train_frames_blob_name);
  const auto cont_blob = net_->blob_by_name(cont_blob_name);
  const auto filter_blob = net_->blob_by_name(filter_blob_name);
  const auto target_blob = net_->blob_by_name(target_blob_name);

  // Randomly select unique episodes to learn from
  std::vector<int> ep_inds(replay_memory_.size());
  std::iota(ep_inds.begin(), ep_inds.end(), 0);
  std::random_shuffle(ep_inds.begin(), ep_inds.end());
  if (ep_inds.size() > kMinibatchSize) {
    ep_inds.resize(kMinibatchSize);
  }


  FramesLayerInputData frame_input;
  TargetLayerInputData target_input;
  FilterLayerInputData filter_input;
  ContLayerInputData cont_input;

  bool active_episodes = ep_inds.size();
  int t = 0;
  int update_step = 0;
  std::vector<std::deque<dqn::FrameDataSp> > past_frames(ep_inds.size());
  while (active_episodes > 0) {
    frame_input.fill(0.0f);
    filter_input.fill(0.0f);
    target_input.fill(0.0f);
    cont_input.fill(1.0f);
    if (t == 0) { // Cont is zeroed for the first step of the episode
      for (int n = 0; n < kMinibatchSize; ++n) {
        cont_input[cont_blob->offset(0,n,0,0)] = 0.0f;
      }
    }
    for (int i = 0; i < kUnroll; ++i, ++t) {
      // FrameVec next_frames;
      // next_frames.reserve(ep_inds.size());
      active_episodes = 0;
      for (int n = 0; n < ep_inds.size(); ++n) {
        const Episode& episode = replay_memory_[ep_inds[n]];
        auto& frame_deque = past_frames[n];
        if (t < episode.size() && std::get<3>(episode[t])) {
          active_episodes++;
          // next_frames.emplace_back(std::get<3>(episode[t]).get());
          frame_deque.push_back(std::get<3>(episode[t]).get());
          while (frame_deque.size() > kInputFramesPerTimestep) {
            frame_deque.pop_front();
          }
        } else {
          frame_deque.clear();
        }
      }
      if (t < kInputFramesPerTimestep) {
        continue;
      }
      // Get the next state QValues
      std::vector<InputFrames> past_frames_vec;
      for (int n = 0; n < ep_inds.size(); ++n) {
        const auto& frame_deque = past_frames[n];
        if (!frame_deque.empty()) {
          CHECK_EQ(frame_deque.size(), kInputFramesPerTimestep);
          InputFrames input_frames;
          std::copy(frame_deque.begin(), frame_deque.end(),
                    input_frames.begin());
          past_frames_vec.emplace_back(input_frames);
        }
      }
      std::vector<ActionValue> actions_and_values =
          SelectActionGreedily(*clone_net_, past_frames_vec, t > 0);
      // Generate the targets/filter/frames inputs
      int target_value_idx = 0;
      for (int n = 0; n < ep_inds.size(); ++n) {
        const Episode& episode = replay_memory_[ep_inds[n]];
        if (t < episode.size()) {
          const auto& transition = episode[t];
          const int action = static_cast<int>(std::get<1>(transition));
          CHECK_LT(action, kOutputCount);
          const float reward = std::get<2>(transition);
          CHECK(reward >= -1.0 && reward <= 1.0);
          const float target = std::get<3>(transition) ?
              reward + gamma_ * actions_and_values[target_value_idx++].second :
              reward;
          // if (std::get<3>(transition)) {
          //   LOG(INFO) << "t: "<< t << " i: " << i << " action: " << action
          //             << " target: " << target << " reward: " << reward
          //             << " maxq: " << actions_and_values[target_value_idx-1].second
          //             << " maxact: " << actions_and_values[target_value_idx-1].first;
          // }
          CHECK(!std::isnan(target));
          filter_input[filter_blob->offset(i,n,action,0)] = 1;
          target_input[target_blob->offset(i,n,action,0)] = target;
          const auto& frame = std::get<0>(transition);
          std::copy(frame->begin(), frame->end(),
                    frame_input.begin() + frames_blob->offset(n,i,0,0));
        }
      }
      CHECK_EQ(target_value_idx, actions_and_values.size());
    }
    InputDataIntoLayers(*net_, frame_input.data(), cont_input.data(),
                        target_input.data(), filter_input.data());
    // LOG(INFO) << "Forward on main net";
    // net_->ForwardPrefilled(nullptr);
    // std::vector<ActionValue> results;
    // results.reserve(ep_inds.size());
    // const auto q_values_blob = net_->blob_by_name("q_values");
    // CHECK(q_values_blob);
    // for (int z = 0; z < 2; ++z) {
    //   for (int i = 0; i < ep_inds.size(); ++i) {
    //     // Get the Q-values from the net: kUnroll*kMinibatchSize*kOutputCount
    //     const auto action_evaluator = [&](Action action) {
    //       const auto q = q_values_blob->data_at(z, i, static_cast<int>(action), 0);
    //       CHECK(!std::isnan(q));
    //       return q;
    //     };
    //     std::vector<float> q_values(legal_actions_.size());
    //     std::transform(legal_actions_.begin(), legal_actions_.end(),
    //                    q_values.begin(), action_evaluator);
    //     PrintQValues(q_values, legal_actions_);
    //   }
    // }
    // exit(0);
    solver_->Step(1);
    update_step++;
  }
  return update_step;
}

int DQN::UpdateRandom() {
  // Every clone_iters steps, update the clone_net_
  if (!clone_net_ || current_iteration() >= last_clone_iter_ + clone_frequency_) {
    LOG(INFO) << "Iter " << current_iteration() << ": Updating Clone Net";
    CloneNet(*test_net_);
    last_clone_iter_ = current_iteration();
  }
  const auto frames_blob = net_->blob_by_name(train_frames_blob_name);
  const auto cont_blob = net_->blob_by_name(cont_blob_name);
  const auto filter_blob = net_->blob_by_name(filter_blob_name);
  const auto target_blob = net_->blob_by_name(target_blob_name);
  FramesLayerInputData frame_input;
  TargetLayerInputData target_input;
  FilterLayerInputData filter_input;
  ContLayerInputData cont_input;
  frame_input.fill(0.0f);
  filter_input.fill(0.0f);
  target_input.fill(0.0f);
  cont_input.fill(1.0f);
  for (int n = 0; n < kMinibatchSize; ++n) {
    cont_input[cont_blob->offset(0,n,0,0)] = 0.0f;
  }
  // Randomly select unique episodes to learn from
  std::vector<int> ep_inds(replay_memory_.size());
  std::iota(ep_inds.begin(), ep_inds.end(), 0);
  std::random_shuffle(ep_inds.begin(), ep_inds.end());
  if (ep_inds.size() > kMinibatchSize) {
    ep_inds.resize(kMinibatchSize);
  }
  int batch_size = ep_inds.size();
  // Time to start each episode
  std::vector<int> ep_starts(batch_size);
  for (int n = 0; n < batch_size; ++n) {
    int ep_size = replay_memory_[ep_inds[n]].size();
    int last_valid_frame = ep_size-1 - (kInputFramesPerTimestep-1) - (kUnroll-1);
    ep_starts[n] = std::uniform_int_distribution<int>
        (0, last_valid_frame)(random_engine);
  }
  for (int u = 0; u < kUnroll; ++u) {
    std::vector<InputFrames> past_frames_vec;
    for (int n = 0; n < batch_size; ++n) {
      const Episode& episode = replay_memory_[ep_inds[n]];
      int last_frame_ts = ep_starts[n] + u + (kInputFramesPerTimestep-1);
      CHECK_GT(episode.size(), last_frame_ts);
      if (std::get<3>(episode[last_frame_ts])) {
        InputFrames input_frames;
        for (int i = 0; i < kInputFramesPerTimestep; ++i) {
          int ts = ep_starts[n] + u + i;
          input_frames[i] = std::get<3>(episode[ts]).get();
        }
        past_frames_vec.push_back(input_frames);
      }
    }
    // Get the next state QValues
    std::vector<ActionValue> actions_and_values =
        SelectActionGreedily(*clone_net_, past_frames_vec, u>0);
    // Generate the targets/filter/frames inputs
    int target_value_idx = 0;
    for (int n = 0; n < batch_size; ++n) {
      const Episode& episode = replay_memory_[ep_inds[n]];
      int ts = ep_starts[n] + u + (kInputFramesPerTimestep-1);
      CHECK_GT(episode.size(), ts);
      const auto& transition = episode[ts];
      const int action = static_cast<int>(std::get<1>(transition));
      CHECK_LT(action, kOutputCount);
      const float reward = std::get<2>(transition);
      CHECK(reward >= -1.0 && reward <= 1.0);
      const float target = std::get<3>(transition) ?
          reward + gamma_ * actions_and_values[target_value_idx++].second :
          reward;
      CHECK(!std::isnan(target));
      filter_input[filter_blob->offset(u,n,action,0)] = 1;
      target_input[target_blob->offset(u,n,action,0)] = target;
      const auto& frame = std::get<0>(transition);
      int frame_idx = u + (kInputFramesPerTimestep-1);
      std::copy(frame->begin(), frame->end(),
                frame_input.begin() + frames_blob->offset(n,frame_idx,0,0));
    }
    CHECK_EQ(target_value_idx, actions_and_values.size());
  }
  // Copy in the pre-input frames
  for (int n = 0; n < batch_size; ++n) {
    for (int i = 0; i < kInputFramesPerTimestep-1; ++i) {
      int ts = ep_starts[n] + i;
      const auto& frame = std::get<0>(replay_memory_[ep_inds[n]][ts]);
      std::copy(frame->begin(), frame->end(),
                frame_input.begin() + frames_blob->offset(n,i,0,0));
    }
  }
  InputDataIntoLayers(*net_, frame_input.data(), cont_input.data(),
                      target_input.data(), filter_input.data());
  solver_->Step(1);
  return 1;
}

void DQN::CloneNet(caffe::Net<float>& net) {
  caffe::NetParameter net_param;
  net.ToProto(&net_param);
  net_param.mutable_state()->set_phase(net.phase());
  if (!clone_net_) {
    clone_net_.reset(new caffe::Net<float>(net_param));
  } else {
    clone_net_->CopyTrainedLayersFrom(net_param);
  }
}

void DQN::InputDataIntoLayers(caffe::Net<float>& net,
                              float* frames_input,
                              float* cont_input,
                              float* target_input,
                              float* filter_input) {
  // Get the layers by name and cast them to memory layers
  const auto frames_input_layer =
      boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
          net.layer_by_name(frames_layer_name));
  const auto cont_input_layer =
      boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
          net.layer_by_name(cont_layer_name));
  const auto target_input_layer =
      boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
          net.layer_by_name(target_layer_name));
  const auto filter_input_layer =
      boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
          net.layer_by_name(filter_layer_name));
  CHECK(frames_input_layer);
  CHECK(cont_input_layer);
  CHECK(target_input_layer);
  CHECK(filter_input_layer);
  frames_input_layer->Reset(frames_input, frames_input,
                            frames_input_layer->batch_size());
  cont_input_layer->Reset(cont_input, cont_input,
                          cont_input_layer->batch_size());
  target_input_layer->Reset(target_input, target_input,
                            target_input_layer->batch_size());
  filter_input_layer->Reset(filter_input, filter_input,
                            filter_input_layer->batch_size());
}
}
