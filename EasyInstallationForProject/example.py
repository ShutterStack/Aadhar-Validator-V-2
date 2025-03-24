import verify

result = verify.aadhar("rahil.jpeg")
if result:
    print("Aadhar card detected!")
else:
    print("Aadhar card not detected.")
