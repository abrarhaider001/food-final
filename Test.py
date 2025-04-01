import requests

def upload_image(image_path):
    url = "https://9805-59-103-106-42.ngrok-free.app/predict"
    
    # Open image file in binary mode
    with open(image_path, 'rb') as img:
        files = {'file': img}  # Assuming the server expects a file with key 'file'
        
        # Send POST request
        response = requests.post(url, files=files)
        
        # Print response
        print("Response Status Code:", response.status_code)
        print("Response JSON:", response.json())

# Example usage
image_path = "C:\\Users\\DELL\\Desktop\\Model\\Image_6.jpg"  # Change this to your image file path
upload_image(image_path)