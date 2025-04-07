import os
import uuid
import random
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from werkzeug.utils import secure_filename
from ml.inference import scan_face_from_array  # Using the function that processes a NumPy image

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a strong secret key

# Hardcoded acne-related products (example products)
ACNE_PRODUCTS = [
    {
        "name": "THE PASTELS SHOP SA Daily Exfoliating Cleanser",
        "ingredients": "Water, Aloe Barbadensis Leaf Extract, Saccharide Isomerate, Glycerin, Cocamidopropyl Betaine, Propylene Glycol, Panthenol, Hydroxyethylcellulose, Phenoxyethanol, Lactic Acid, Sodium Chloride, Camellia Sinensis Leaf Extract, Sodium Acetate, Ethylhexylglycerin, Salicyclic Acid, Dioscorea Villosa (Wild Yam) Root Extract, Potassium Sorbate, Sodium Benzoate, Cellulose, Dextrin, Polydextrose, Amylopectin, Niacinamide.",
        "disclaimer": "Avoid if allergic to salicylic acid. Safe for pregnant women.",
        "image": "placeholder1.jpg",
        "shopee_link": "#",
        "lazada_link": "#"
    },
    {
        "name": "DAUGHTERS OF MALAYA Blemish-Free Jelly Cleanser",
        "ingredients": "Water, Ammonium Lauryl Sulfate, Lauryl Glucoside, Glycerin, Palm Kernelamide Dea, Acrylates/C10-30 Alkyl Acrylate Crosspolymer, Centella Asiatica Leaf Extract, Chamomilla Recutita (Matricaria) Flower Extract, Salicylic Acid, Calendula Officinalis Flower Extract, Gluconolactone, Hydroxyethylcellulose, Polyquaternium-7, Panthenol, Lactobacillus Ferment, Maltodextrin, Phenoxyethanol, Ethylhexylglycerin.",
        "disclaimer": "Not recommended for sensitive skin. Consult a doctor if pregnant.",
        "image": "placeholder2.jpg",
        "shopee_link": "#",
        "lazada_link": "#"
    },
    {
        "name": "Bougas Beauty Daily Moisturiser",
        "ingredients": "Rosa Damascena Flower Water, Hyaluronic Acid, Bee Venom, Symphytum Officinale Leaf Extract, Niacinamide, Imperata Cylindrica Root Extract, Glycyrrhiza Glabra (Licorice) Root Extract",
        "disclaimer": "Contains bee venom, which may not be suitable for individuals with allergies. Pregnant women should consult a healthcare provider before use.",
        "image": "placeholder3.jpg",
        "shopee_link": "#",
        "lazada_link": "#"
    },
    {
        "name": "Ruruberry 10% Niacinamide + HA",
        "ingredients": "Water, Niacinamide, Butylene Glycol, Acetyl Glucosamine, Xylitylglucoside, Hydrolyzed Soy Protein, Anhydroxylitol, Sodium Hyaluronate, Sodium Hyaluronate Crosspolymer, Sodium Acetylated Hyaluronate, Sodium Carboxymethyl Beta-Glucan, PPG-20 Methyl Glucose Ether, Xylitol, Tetrasodium EDTA, Sodium Ascorbyl Phosphate, Panthenol, Citric Acid, Pentylene Glycol, Centella Asiatica Extract, Glycerin, Phenoxyethanol, Chlorphenesin, Polygonum Cuspidatum Root Extract, Scutellaria Baicalensis Root Extract, Camellia Sinensis Leaf Extract, Glycyrrhiza Glabra (Licorice) Root Extract, Chamomilla Recutita (Matricaria) Flower Extract, Rosmarinus Officinalis (Rosmary) Leaf Extract, Hydrolyzed Sodium Hyaluronate, Ethylhexylglycerin",
        "disclaimer": "No major allergens. Safe for pregnant women.",
        "image": "placeholder4.jpg",
        "shopee_link": "#",
        "lazada_link": "#"
    },
    {
        "name": "ZarZou Repairing and Soothing Facial Serum",
        "ingredients": "Aqua, Hyaluronic Acid, Hamamelis Virginiana (Witch Hazel) Water, Carbomer, Niacinamide, Centella Asiatica (Centella) Extract, Glycerin, Tocopheryl Acetate, Rosa Moschata (Rosehip) Seed Oil, Arginine, Rosa Damascena (Damask Rose) Flower Oil, Phenoxyethanol, Caprylyl Glycol, Melaleuca Alternifolia (Tea Tree) Leaf Oil, Squalane, Aloe Barbadensis (Aloe Vera) Leaf Juice, Maltodextrin, Fucus Vesiculosus Extract, Laminaria Digitata Extract, Spirulina Maxima Extract, Porphyra Umbilicalis Extract, Ascophyllum Nodosum Extract",
        "disclaimer": "No major allergens. Safe for pregnant women.",
        "image": "placeholder5.jpg",
        "shopee_link": "#",
        "lazada_link": "#"
    },
    {
        "name": "Kayman Beauty Rosa Glow Treatment Essence",
        "ingredients": "Water, Cucumber Extract, Glycerine, Beetox-H, Xanthan Gum, Rosa Damascena Flower Water, Alpha Arbutin, Phenoxyethanol",
        "disclaimer": "Do a patch test before use. Safe for pregnant women.",
        "image": "placeholder6.jpg",
        "shopee_link": "#",
        "lazada_link": "#"
    },
    {
        "name": "KAYMAN BEAUTY Suntella UV Milk SPF50+ PA++++",
        "ingredients": "Water, Glycerin, Dibutyl Adipate, Dipropylene Glycol, Butyloctyl Salicylate, Titanium Dioxide (CI 77891), Isononyl Isononanoate, Polyglyceryl-6 Stearate, Niacinamide, Diethylamino Hydroxybenzoyl Hexyl Benzoate, Phenethyl Benzoate, 1,2-Hexanediol, Bis-Ethylhexyloxyphenol, Methoxyphenyl Triazine, Ethylhexyl Triazone, Polysilicone-15, Terephthalylidene Dicamphor Sulfonic Acid, Zinc Oxide (CI 77947), Synthetic Fluorphlogopite, Polyglyceryl-3 Distearate, Potassium Cetyl Phosphate, Polyhydroxystearic Acid, Centella Asiatica Extract, Tromethamine, Sodium Acrylates Crosspolymer-2, Hydroxyethyl Acrylate/Sodium Acryloyldimethyl Taurate Copolymer, Caprylic/Capric Triglyceride, Aluminum Hydroxide, Acrylates/C10-30 Alkyl Acrylate Crosspolymer, Hydroxyacetophenone, Stearic Acid, Polyglyceryl-6 Behenate, Hydrogenated C6-20 Polyolefin, Allantoin, Polyglyceryl-2 Dipolyhydroxystearate, Glyceryl Stearate Citrate, Adenosine, Trisodium Ethylenediamine Disuccinate, HDI/Trimethylol Hexyllactone Crosspolymer, Dimethicone, Silica, Panthenol, Hydrogenated Lecithin, Ceramide NP",
        "disclaimer": "No major allergens. Safe for pregnant women.",
        "image": "placeholder7.jpg",
        "shopee_link": "#",
        "lazada_link": "#"
    },
    {
        "name": "Bougas Beauty Hydrating Sunscreen",
        "ingredients": "Water, Ethylhexyl Methoxycin-namate, Butyl Methoxydibenzoylmeth-ane, Octocrylene, Phospholipids, Butylene Glycol, Glycerin, Cetyl Alcohol, Caprylic/Capric Triglyceride, Titanium Dioxide, C12-15 Alkyl Benzoate, Isododecane, Niacinamide, Phenoxyethanol, Caprylyl Glycol, Tocopheryl Acetate, Sodium Hyaluronate, Aloe Barbadensis Leaf Extract, Butylene Glycol, Anthemis Nobilis Flower Extract, Allantoin, Xanthan Gum, Bisabolol.",
        "disclaimer": "No major allergens. Safe for pregnant women.",
        "image": "placeholder8.jpg",
        "shopee_link": "#",
        "lazada_link": "#"
    }
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/premium')
def premium():
    return render_template('premium.html')

@app.route('/results')
def results():
    condition = request.args.get('condition', 'Unknown')
    img_file = request.args.get('img_file', None)
    
    prod_list = ACNE_PRODUCTS.copy()
    for prod in prod_list:
        prod['rating'] = round(random.uniform(3.5, 5.0), 1)
    random.shuffle(prod_list)
    featured = prod_list[:3]
    others = prod_list[3:]
    
    return render_template('results.html', condition=condition, img_file=img_file, featured=featured, others=others)

@app.route('/scan', methods=['POST'])
def scan():
    try:
        # Check if a file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image part'}), 400
        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Read the image file bytes
        file_bytes = file.read()

        # Check if the file is not empty
        if not file_bytes:
            return jsonify({'error': 'Empty file'}), 400

        # Decode the image
        image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400

        # Run inference to get the predicted skin condition
        prediction = scan_face_from_array(image)

        # Generate a unique filename and save the image
        filename = secure_filename(str(uuid.uuid4()) + ".jpg")
        folder = os.path.join("static", "scanned_images")
        os.makedirs(folder, exist_ok=True)
        image_path = os.path.join(folder, filename)
        cv2.imwrite(image_path, image)

        # Return JSON response with the condition and image filename
        return jsonify({'condition': prediction, 'img_file': filename})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred while processing the image'}), 500
        
@app.route('/subscribe')
def subscribe():
    return render_template('subscribe.html')

@app.route('/dietary')
def dietary():
    return render_template('dietary.html')

@app.route('/progress')
def progress():
    # Get skin condition from session or set a default if not available
    skin_condition = session.get('skin_condition', {
      "Papules & Pustules": 0.65,
      "Black & Whitehead": 0.45,
      "Cyst": 0.10,
      "Acne": 0.80
    })

    # Get the image file path from session
    img_file = session.get('img_file')

    # Render the progress page with skin condition and image file
    return render_template('progress.html', skin_condition=skin_condition, img_file=img_file)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        if '@' in email:
            session['logged_in'] = True
            session['email'] = email
            return redirect(url_for('premium'))
        else:
            return "Invalid email/password. Please try again."

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    # Retrieve user information from the session
    user_name = session.get('user_name', 'Jane Doe')
    user_email = session.get('email')
    user_age = session.get('user_age', 25)  # Default age for now
    user_address = session.get('user_address', '123 Street, City, Country')  # Default address for now

    if request.method == 'POST':
        # Update user information in the session
        session['user_name'] = request.form['name']
        session['user_age'] = request.form['age']
        session['user_address'] = request.form['address']
        
        return redirect(url_for('profile'))

    user = {
        "name": user_name,
        "email": user_email,
        "age": user_age,
        "address": user_address,
        "purchase_history": [
            {"date": "2025-03-10", "product": "THE PASTELS SHOP SA Daily Exfoliating Cleanser", "image": "placeholder1.jpg", "rating": 4.5},
            {"date": "2025-03-12", "product": "DAUGHTERS OF MALAYA Blemish-Free Jelly Cleanser", "image": "placeholder2.jpg", "rating": 4.3},
            {"date": "2025-03-14", "product": "Bougas Beauty Hydrating Sunscreen", "image": "placeholder8.jpg", "rating": 4.7}
        ]
    }

    return render_template('profile.html', user=user)


if __name__ == '__main__':
    app.run(debug=True)
