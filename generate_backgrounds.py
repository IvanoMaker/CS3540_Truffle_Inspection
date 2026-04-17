import os
import random
from PIL import Image, ImageDraw

def generate_background_image(width=224, height=224):
    """Generate a synthetic background image resembling a white conveyor belt with pink chocolate drizzles, metallic grays, and occasional blue sections."""
    # Create base white image with slight texture variation
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Add subtle texture to the conveyor belt (slight gray variations)
    for _ in range(width * height // 100):  # Add some noise
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        gray = random.randint(240, 255)
        draw.point((x, y), (gray, gray, gray))

    # Pink color palette for chocolate drizzles
    pink_colors = [
        (255, 192, 203),  # Pink
        (255, 20, 147),   # Deep Pink
        (255, 105, 180),  # Hot Pink
        (255, 182, 193),  # Light Pink
        (219, 112, 147),  # Pale Violet Red
        (255, 160, 122),  # Light Salmon (for variation)
    ]

    # Add metallic gray sections (for machinery/conveyor parts)
    num_gray_sections = random.randint(1, 3)
    for _ in range(num_gray_sections):
        gray_color = random.choice([(128, 128, 128), (169, 169, 169), (192, 192, 192), (211, 211, 211)])  # Metallic grays
        x1 = random.randint(0, width // 2)
        y1 = random.randint(0, height // 2)
        x2 = x1 + random.randint(20, 60)
        y2 = y1 + random.randint(20, 60)
        x2 = min(x2, width)
        y2 = min(y2, height)
        draw.rectangle((x1, y1, x2, y2), fill=gray_color)

    # Add occasional blue sections (for nitrile gloves)
    if random.random() < 0.4:  # 40% chance
        num_blue_sections = random.randint(1, 2)
        for _ in range(num_blue_sections):
            blue_color = random.choice([(0, 0, 255), (0, 191, 255), (30, 144, 255), (70, 130, 180)])  # Blue shades
            x1 = random.randint(width // 4, width - 50)
            y1 = random.randint(height // 4, height - 50)
            x2 = x1 + random.randint(15, 40)
            y2 = y1 + random.randint(15, 40)
            x2 = min(x2, width)
            y2 = min(y2, height)
            draw.rectangle((x1, y1, x2, y2), fill=blue_color)

    # Add pink chocolate drizzles (random lines and blobs)
    num_drizzles = random.randint(3, 8)
    for _ in range(num_drizzles):
        color = random.choice(pink_colors)
        # Random line drizzle
        start_x = random.randint(0, width)
        start_y = random.randint(0, height)
        end_x = start_x + random.randint(-50, 50)
        end_y = start_y + random.randint(10, 80)  # Drizzles tend downward
        end_x = max(0, min(width, end_x))
        end_y = max(0, min(height, end_y))
        draw.line((start_x, start_y, end_x, end_y), fill=color, width=random.randint(2, 6))

        # Add some blob drizzles
        if random.random() < 0.5:
            blob_x = random.randint(0, width)
            blob_y = random.randint(0, height)
            blob_size = random.randint(5, 15)
            draw.ellipse((blob_x-blob_size, blob_y-blob_size, blob_x+blob_size, blob_y+blob_size), fill=color)

    return img

def generate_background_images(num_images=100, output_dir='dataset/train/truffles/', prefix='background'):
    """Generate and save background images to the specified directory."""
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_images):
        img = generate_background_image()
        filename = f"{prefix}_{i+1:03d}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        print(f"Generated: {filepath}")

if __name__ == "__main__":
    # Generate 100 background images for training
    print("Generating 100 background images for training...")
    generate_background_images(100, 'dataset/train/truffles/', 'background_train')

    # Generate 100 background images for validation
    print("Generating 100 background images for validation...")
    generate_background_images(100, 'dataset/val/truffles/', 'background_val')

    print("Background image generation complete!")