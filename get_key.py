import oci
from cryptography.hazmat.primitives import serialization

# Path to your existing private key
private_key_path = r"C:\Users\ashir\Downloads\AI Mining Projects Recent FIles\ashirkakhter@gmail.com-2026-03-18T06_03_56.362Z.pem"

with open(private_key_path, "rb") as key_file:
    # Load the private key
    private_key = serialization.load_pem_private_key(
        key_file.read(),
        password=None,
    )

# Extract the public key in PEM format
public_key = private_key.public_key().public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

# Print the result
print(public_key.decode("utf-8"))

# Save it to a file
with open("oci_api_key_public.pem", "wb") as f:
    f.write(public_key)
