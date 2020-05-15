import rsa

def main():
    pub_key, prv_key = rsa.generateKeys()
    print("public key: " + str(pub_key))
    print("private key: " + str(prv_key))

    return

if __name__ == "__main__":
    main()