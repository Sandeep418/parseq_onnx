#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <filesystem>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>


// TokenDecoder class
class TokenDecoder {
public:
    TokenDecoder() {
        // Initialize special tokens and charset
        specials_first = { "[E]" };
        specials_last = { "[B]", "[P]" };
        charset = {
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
            "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
            "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "!", "\"", "#", "$", "%", "&", "'", "(", ")",
            "*", "+", ",", "-", ".", "/", ":", ";", "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~", "₹", "¥", " "
        };

        // Build itos and stoi maps
        itos.insert(itos.end(), specials_first.begin(), specials_first.end());
        itos.insert(itos.end(), charset.begin(), charset.end());
        itos.insert(itos.end(), specials_last.begin(), specials_last.end());

        for (size_t i = 0; i < itos.size(); ++i) {
            stoi[itos[i]] = static_cast<int>(i);
        }
    }

    std::string ids2tok(const std::vector<int>& token_ids, bool join = true) {
        std::string result;
        for (int id : token_ids) {
            if (id >= 0 && id < static_cast<int>(itos.size())) {
                result += itos[id];
            }
        }
        return result;
    }

    std::pair<std::vector<float>, std::vector<int>> filter(const std::vector<float>& probs, const std::vector<int>& ids) {
        int eos_id = stoi[specials_first[0]]; // [E]

        // Find the position of EOS token
        auto eos_it = std::find(ids.begin(), ids.end(), eos_id);
        size_t eos_idx = eos_it != ids.end() ? std::distance(ids.begin(), eos_it) : ids.size();

        // Filter ids up to EOS token
        std::vector<int> filtered_ids(ids.begin(), ids.begin() + eos_idx);

        // Filter probs up to and including EOS token (if present)
        size_t probs_size = std::min(probs.size(), eos_idx + 1);
        std::vector<float> filtered_probs(probs.begin(), probs.begin() + probs_size);

        return { filtered_probs, filtered_ids };
    }

    std::pair<std::vector<std::string>, std::vector<std::vector<float>>> decode(const std::vector<std::vector<std::vector<float>>>& token_dists, bool raw = false) {
        std::vector<std::string> batch_tokens;
        std::vector<std::vector<float>> batch_probs;

        for (const auto& dist : token_dists) {
            std::vector<float> probs;
            std::vector<int> ids;

            // For each position in the sequence
            for (const auto& pos_dist : dist) {
                // Find the max probability and its index (greedy selection)
                auto max_it = std::max_element(pos_dist.begin(), pos_dist.end());
                float max_prob = *max_it;
                int max_id = static_cast<int>(std::distance(pos_dist.begin(), max_it));

                probs.push_back(max_prob);
                ids.push_back(max_id);
            }

            if (!raw) {
                auto result = filter(probs, ids);
                probs = result.first;
                ids = result.second;
            }

            std::string token = ids2tok(ids);
            batch_tokens.push_back(token);
            batch_probs.push_back(probs);
        }

        return { batch_tokens, batch_probs };
    }

private:
    std::vector<std::string> specials_first;
    std::vector<std::string> specials_last;
    std::vector<std::string> charset;
    std::vector<std::string> itos;
    std::unordered_map<std::string, int> stoi;
};

// Utility function to apply softmax to a vector
std::vector<float> softmax(const std::vector<float>& input) {
    float max_val = *std::max_element(input.begin(), input.end());
    std::vector<float> exp_values;
    float sum_exp = 0.0f;

    for (float val : input) {
        float exp_val = std::exp(val - max_val); // Subtract max for numerical stability
        exp_values.push_back(exp_val);
        sum_exp += exp_val;
    }

    // Normalize
    for (float& val : exp_values) {
        val /= sum_exp;
    }

    return exp_values;
}

// Function to preprocess image for model input
std::vector<float> preprocess_image(const cv::Mat& image) {
    // Resize to 32x128
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(128, 32));

    // Convert to float and normalize to [-1, 1]
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32F, 1.0 / 255.0);
    float_img = (float_img - 0.5f) / 0.5f;

    // Rearrange to CHW format (3x32x128)
    std::vector<float> tensor_values(3 * 32 * 128);
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < 32; h++) {
            for (int w = 0; w < 128; w++) {
                tensor_values[c * 32 * 128 + h * 128 + w] = float_img.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    return tensor_values;
}

int main() {
    try {
        // Initialize ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Parseq");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Load model
        std::wstring model_path = "parseq_trained.onnx";
        Ort::Session session(env, model_path.c_str(), session_options);
        std::cout << "✅ Model Loaded Successfully on CPU!\n";

        // TokenDecoder instance
        TokenDecoder decoder;

        // Get input and output names
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = session.GetInputNameAllocated(0, allocator);
        auto output_name = session.GetOutputNameAllocated(0, allocator);

        // Directory containing images
        std::string img_dir = "D:/OCR_DATA/Benchmark_A1/";
        for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
            std::string img_name = entry.path().filename().string();
            std::string image_path = img_dir + img_name;
            std::cout << "Processing: " << img_name << std::endl;

            // Load image
            cv::Mat original_image = cv::imread(image_path);
            if (original_image.empty()) {
                std::cerr << "Failed to load image: " << img_name << std::endl;
                continue;
            }

            // Keep original for display
            cv::Mat display_image = original_image.clone();

            // Preprocess image
            std::vector<float> input_tensor_values = preprocess_image(original_image);

            // Create input tensor
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<int64_t> input_shape = { 1, 3, 32, 128 };  // NCHW format
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, input_tensor_values.data(), input_tensor_values.size(),
                input_shape.data(), input_shape.size()
            );

            // Run inference
            Ort::RunOptions run_options;
            std::vector<const char*> input_names = { input_name.get() };
            std::vector<const char*> output_names = { output_name.get() };

            auto start_time = std::chrono::high_resolution_clock::now();
            auto output_tensors = session.Run(
                run_options, input_names.data(), &input_tensor, 1,
                output_names.data(), 1
            );
            auto end_time = std::chrono::high_resolution_clock::now();

            // Get output tensor info
            auto shape_info = output_tensors[0].GetTensorTypeAndShapeInfo();
            auto dimensions = shape_info.GetShape();
            float* output_data = output_tensors[0].GetTensorMutableData<float>();

            // Extract dimensions
            int64_t batch_size = dimensions[0];
            int64_t seq_length = dimensions[1];
            int64_t vocab_size = dimensions[2];

            std::cout << "Output shape: [" << batch_size << ", " << seq_length << ", " << vocab_size << "]" << std::endl;

            // Reshape and apply softmax
            std::vector<std::vector<std::vector<float>>> softmax_output(batch_size);
            for (int64_t b = 0; b < batch_size; b++) {
                softmax_output[b].resize(seq_length);
                for (int64_t s = 0; s < seq_length; s++) {
                    // Extract distribution for current token
                    std::vector<float> token_logits(vocab_size);
                    for (int64_t v = 0; v < vocab_size; v++) {
                        token_logits[v] = output_data[b * seq_length * vocab_size + s * vocab_size + v];
                    }

                    // Apply softmax
                    softmax_output[b][s] = softmax(token_logits);
                }
            }

            // Decode predictions
            auto result = decoder.decode(softmax_output);
            auto predictions = result.first;
            auto confidence_scores = result.second;

            // Print results
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            std::cout << "Time Taken: " << duration << " ms" << std::endl;

            if (!predictions.empty()) {
                std::cout << "Prediction: " << predictions[0] << std::endl;

                if (!confidence_scores.empty()) {
                    // Calculate average confidence
                    float avg_conf = 0.0f;
                    if (!confidence_scores[0].empty()) {
                        avg_conf = std::accumulate(confidence_scores[0].begin(), confidence_scores[0].end(), 0.0f) /
                            confidence_scores[0].size();
                    }

                    for (int i = 0;i < confidence_scores[0].size();i++)
                    {
                        std::cout<<confidence_scores[0][i]<<" ";
                    }

                   
                    std::cout << "Average Confidence: " << avg_conf << std::endl;
                }
            }

            std::cout << "------------------------------" << std::endl;

            // Display image
            cv::imshow("Image", display_image);
            cv::waitKey(0);
        }
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
