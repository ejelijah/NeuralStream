import os

class Note:
    def __init__(self, title, content):
        self.title = title
        self.content = content
        self.file_path = os.path.join("notes", f"{title}.txt")

    def save(self):
        with open(self.file_path, "w") as f:
            f.write(self.content)

    def load(self):
        with open(self.file_path, "r") as f:
            self.content = f.read()