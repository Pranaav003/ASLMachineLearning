# 5. Test the Model
# Run this file to test the model in real-time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from preprocess_3 import X_test, actions
from init_1 import mediapipe_detection, draw_styled_landmarks, extract_keypoints
import mediapipe as mp
from tensorflow.keras.models import load_model

mp_holistic = mp.solutions.holistic

num_symbols = len(actions)  # Ensure this matches your actions
colormap = plt.cm.get_cmap('hsv', num_symbols)
colors = [colormap(i)[:3] for i in range(num_symbols)]
colors = [(int(r*255), int(g*255), int(b*255)) for r, g, b in colors]

def prob_viz(res, actions, input_frame, colors):
    """
    Visualizes prediction probabilities on the frame.

    Args:
        res (list): List of probabilities from the model.
        actions (list): List of action labels.
        input_frame (numpy.ndarray): The current video frame.
        colors (list): List of colors for each action.

    Returns:
        numpy.ndarray: The frame with visualization added.
    """
    output_frame = input_frame.copy()
    
    # Ensure res, actions, and colors are of the same length
    if len(res) != len(actions):
        print("Error: The number of probabilities does not match the number of actions.")
        return output_frame

    for num, prob in enumerate(res):
        if num >= len(colors):  # Prevent index out of range
            print(f"Warning: Color index {num} exceeds available colors. Skipping visualization.")
            continue
        
        cv2.rectangle(output_frame, (0, 60 + num * 40),
                      (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, f'{actions[num]}: {prob:.2f}', (0, 85 + num * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame


def run_test_model():
    # 1. Detection Variables
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.4  # Updated to match the notebook

    cap = cv2.VideoCapture(0)

    # Load the trained model
    model = load_model('final_model.keras')  # Ensure this path is correct

    print(f"Actions: {actions}")  # Debugging: Print actions list

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.4
    ) as holistic:
        while cap.isOpened():
            # Read Feed
            ret, frame = cap.read()
            if not ret:
                break

            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            # Draw Landmarks
            draw_styled_landmarks(image, results)

            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]  # Ensure sequence length matches the notebook

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(f"Prediction probabilities: {res}")  # Debugging: Print model output
                if len(res) != len(actions):
                    print(f"Mismatch: Model output size {len(res)} vs actions size {len(actions)}")
                    break  # Exit if there's a mismatch to avoid further errors

                predicted_index = np.argmax(res)
                if predicted_index >= len(actions):
                    print(f"Predicted index {predicted_index} is out of bounds for actions list.")
                    break  # Exit to prevent further issues

                print(f"Predicted action: {actions[predicted_index]}")
                predictions.append(predicted_index)

                # 3. Visualization logic
                if len(predictions) >= 10:
                    recent_predictions = predictions[-10:]
                    unique_predictions = np.unique(recent_predictions)
                    if len(unique_predictions) == 1 and unique_predictions[0] == predicted_index:
                        if res[predicted_index] > threshold:
                            if len(sentence) > 0:
                                if actions[predicted_index] != sentence[-1]:
                                    sentence.append(actions[predicted_index])
                            else:
                                sentence.append(actions[predicted_index])
                    # Move this outside the threshold block
                    if len(sentence) > 5:
                        sentence = sentence[-5:]

                # Visualization Probabilities
                image = prob_viz(res, actions, image, colors)

            # Get image dimensions
            height, width, _ = image.shape

            # Display the output as subtitles
            rectangle_bgr = (0, 0, 0)  # Black rectangle for subtitles
            cv2.rectangle(image, (0, height - 100), (width, height), rectangle_bgr, -1)

            # Add text over the rectangle
            cv2.putText(
                image,
                ' '.join(sentence),
                (10, height - 10),  # Position text slightly above the bottom
                cv2.FONT_HERSHEY_SIMPLEX,
                3,  # Font scale
                (255, 255, 255),  # White text
                3,  # Thickness
                cv2.LINE_AA
            )

            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

# Only run the test if this script is executed directly
if __name__ == "__main__":
    run_test_model()
