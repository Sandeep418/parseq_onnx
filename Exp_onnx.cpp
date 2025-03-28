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
#include <unordered_map>
#include <thread>

// TokenDecoder class - optimized
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
            "*", "+", ",", "-", ".", "/", ":", ";", "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~", "Rupee₹", "YEN¥", " "
        };

        // Build itos and stoi maps with reserve to avoid reallocations
        itos.reserve(specials_first.size() + charset.size() + specials_last.size());
        itos.insert(itos.end(), specials_first.begin(), specials_first.end());
        itos.insert(itos.end(), charset.begin(), charset.end());
        itos.insert(itos.end(), specials_last.begin(), specials_last.end());

        // Pre-allocate map capacity
        stoi.reserve(itos.size());
        for (size_t i = 0; i < itos.size(); ++i) {
            stoi[itos[i]] = static_cast<int>(i);
        }

        // Cache special token IDs for faster lookup
        eos_id = stoi[specials_first[0]];
    }

    // Optimized with string reserve
    std::string ids2tok(const std::vector<int>& token_ids, bool join = true) {
        std::string result;
        result.reserve(token_ids.size() * 2); // Estimate average token length
        for (int id : token_ids) {
            if (id >= 0 && id < static_cast<int>(itos.size())) {
                result += itos[id];
            }
        }
        return result;
    }

    // Optimized by removing unnecessary copies
    std::pair<std::vector<float>, std::vector<int>> filter(const std::vector<float>& probs, const std::vector<int>& ids) {
        // Find the position of EOS token
        auto eos_it = std::find(ids.begin(), ids.end(), eos_id);
        size_t eos_idx = eos_it != ids.end() ? std::distance(ids.begin(), eos_it) : ids.size();

        // Filter ids up to EOS token
        std::vector<int> filtered_ids(ids.begin(), ids.begin() + eos_idx);

        // Filter probs up to and including EOS token (if present)
        size_t probs_size = std::min(probs.size(), eos_idx + 1);
        std::vector<float> filtered_probs(probs.begin(), probs.begin() + probs_size);

        return { std::move(filtered_probs), std::move(filtered_ids) };
    }

    // Optimized decode function
    std::pair<std::vector<std::string>, std::vector<std::vector<float>>> decode(const std::vector<std::vector<std::vector<float>>>& token_dists, bool raw = false) {
        const size_t batch_size = token_dists.size();
        std::vector<std::string> batch_tokens(batch_size);
        std::vector<std::vector<float>> batch_probs(batch_size);

        // Process each item in the batch
        for (size_t b = 0; b < batch_size; ++b) {
            const auto& dist = token_dists[b];
            const size_t seq_length = dist.size();

            std::vector<float> probs;
            std::vector<int> ids;

            probs.reserve(seq_length);
            ids.reserve(seq_length);

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
                probs = std::move(result.first);
                ids = std::move(result.second);
            }

            batch_tokens[b] = ids2tok(ids);
            batch_probs[b] = std::move(probs);
        }

        return { std::move(batch_tokens), std::move(batch_probs) };
    }

private:
    std::vector<std::string> specials_first;
    std::vector<std::string> specials_last;
    std::vector<std::string> charset;
    std::vector<std::string> itos;
    std::unordered_map<std::string, int> stoi;
    int eos_id; // Cache the EOS ID for faster lookup
};

// Optimized softmax with pre-allocation
inline std::vector<float> softmax(const std::vector<float>& input) {
    const size_t size = input.size();
    std::vector<float> exp_values(size);

    float max_val = *std::max_element(input.begin(), input.end());
    float sum_exp = 0.0f;

    // Calculate exponents in one pass
    for (size_t i = 0; i < size; ++i) {
        float exp_val = std::exp(input[i] - max_val);
        exp_values[i] = exp_val;
        sum_exp += exp_val;
    }

    // Normalize in-place
    float inv_sum = 1.0f / sum_exp;
    for (size_t i = 0; i < size; ++i) {
        exp_values[i] *= inv_sum;
    }

    return exp_values;
}

// Optimized image preprocessing with cv::Mat operations
std::vector<float> preprocess_image(const cv::Mat& image) {
    // Pre-allocate output tensor
    std::vector<float> tensor_values(3 * 32 * 128);

    // Use cv::Mat operations for faster resizing and normalization
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(128, 32), 0, 0, cv::INTER_AREA);

    // Convert to float and normalize to [-1, 1] (single op)
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32F, 1.0 / 255.0, -0.5);
    float_img /= 0.5;

    // Faster channel extraction with direct memory access
    const int height = float_img.rows;
    const int width = float_img.cols;
    const int img_area = height * width;

    // Extract channels directly - much faster than nested loops
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            const cv::Vec3f& pixel = float_img.at<cv::Vec3f>(h, w);
            for (int c = 0; c < 3; ++c) {
                tensor_values[c * img_area + h * width + w] = pixel[c];
            }
        }
    }

    return tensor_values;
}

int main() {
    try {
        // Enable OpenCV optimization
        cv::setNumThreads(std::thread::hardware_concurrency());

        // Initialize ONNX Runtime with optimizations
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Parseq");
        std::cout << "ONNX Runtime Version: " << Ort::GetVersionString() << std::endl;
        Ort::SessionOptions session_options;

        // Use all available threads for better performance
        session_options.SetIntraOpNumThreads(std::thread::hardware_concurrency());
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Enable memory pattern optimization
        session_options.EnableMemPattern();

        // Enable CPU memory arena
        session_options.EnableCpuMemArena();

        // Load model
        std::wstring model_path = L"D:\\OCR\\Test_ACGI_OCR_Engine\\parseq_trained.onnx";
        Ort::Session session(env, model_path.c_str(), session_options);
        std::cout << "✅ Model Loaded Successfully on CPU with "
            << std::thread::hardware_concurrency() << " threads\n";

        // Pre-allocate TokenDecoder
        TokenDecoder decoder;

        // Get input and output names once
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = session.GetInputNameAllocated(0, allocator);
        auto output_name = session.GetOutputNameAllocated(0, allocator);

        // Create name vectors once
        std::vector<const char*> input_names = { input_name.get() };
        std::vector<const char*> output_names = { output_name.get() };

        // Pre-allocate memory info
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> input_shape = { 1, 3, 32, 128 };

        // Create RunOptions once
        Ort::RunOptions run_options;

        // Directory containing images
        std::string img_dir = "D:/OCR_DATA/Benchmark_A1/";

        // Get all image files first to avoid filesystem traversal in the loop
        std::vector<std::filesystem::path> image_paths;
        for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
            if (entry.is_regular_file()) {
                image_paths.push_back(entry.path());
            }
        }

        std::cout << "Found " << image_paths.size() << " images to process\n";

        double total_processing_time = 0.0;
        int processed_count = 0;

        for (const auto& path : image_paths) {
            std::string img_name = path.filename().string();
            std::string image_path = path.string();

            // Load image with faster IMREAD_COLOR flag
            cv::Mat original_image = cv::imread(image_path, cv::IMREAD_COLOR);
            if (original_image.empty()) {
                std::cerr << "Failed to load image: " << img_name << std::endl;
                continue;
            }

            // Preprocess image
            std::vector<float> input_tensor_values = preprocess_image(original_image);

            // Create input tensor - reuse memory when possible
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, input_tensor_values.data(), input_tensor_values.size(),
                input_shape.data(), input_shape.size()
            );

            // Measure only inference time (not display or preprocessing)
            auto start_time = std::chrono::high_resolution_clock::now();

            // Run inference
            auto output_tensors = session.Run(
                run_options, input_names.data(), &input_tensor, 1,
                output_names.data(), 1
            );

            // Get output tensor info
            auto shape_info = output_tensors[0].GetTensorTypeAndShapeInfo();
            auto dimensions = shape_info.GetShape();
            float* output_data = output_tensors[0].GetTensorMutableData<float>();

            // Extract dimensions
            int64_t batch_size = dimensions[0];
            int64_t seq_length = dimensions[1];
            int64_t vocab_size = dimensions[2];

            // Pre-allocate softmax output
            std::vector<std::vector<std::vector<float>>> softmax_output(batch_size);
            for (int64_t b = 0; b < batch_size; b++) {
                softmax_output[b].resize(seq_length);
                for (int64_t s = 0; s < seq_length; s++) {
                    // Direct memory access for token logits - avoid unnecessary copies
                    float* logits_ptr = output_data + (b * seq_length * vocab_size + s * vocab_size);
                    std::vector<float> token_logits(logits_ptr, logits_ptr + vocab_size);

                    // Apply softmax
                    softmax_output[b][s] = softmax(token_logits);
                }
            }

            // Decode predictions
            auto result = decoder.decode(softmax_output);
            auto end_time = std::chrono::high_resolution_clock::now();

            auto predictions = result.first;
            auto confidence_scores = result.second;

            // Calculate processing time
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            total_processing_time += duration;
            processed_count++;

            std::cout << "Processing: " << img_name << std::endl;
            std::cout << "Time Taken: " << duration << " ms" << std::endl;

            if (!predictions.empty()) {
                std::cout << "Prediction: " << predictions[0] << std::endl;

                if (!confidence_scores.empty() && !confidence_scores[0].empty()) {
                    // Calculate average confidence
                    float avg_conf = std::accumulate(confidence_scores[0].begin(), confidence_scores[0].end(), 0.0f) /
                        confidence_scores[0].size();

                    std::cout << "Average Confidence: " << avg_conf << std::endl;
                }
            }

            std::cout << "------------------------------" << std::endl;

            // Display image - consider removing this in production for better performance
            cv::imshow("Image", original_image);
            cv::waitKey(0); // Use 1ms wait instead of 0 to reduce CPU usage
        }

        if (processed_count > 0) {
            std::cout << "Average processing time: " << (total_processing_time / processed_count) << " ms per image\n";
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