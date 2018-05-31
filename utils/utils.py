class Dataset():
    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        self.categrade_info = []
        self.source_categrade_ids = {}
        self.class_info = []
        self.class_categrade_ids = {}

    def add_categrade(self, source, categrade_id, categrade_name,class_num):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.categrade_info:
            if info['source'] == source and info["id"] == categrade_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.categrade_info.append({
            "source": source,
            "id": categrade_id,
            "name": categrade_name,
            "classnum": class_num
        })

    def add_classes(self,categrade,class_id,class_name):
        for info in self.categrade_info:
            if info['source'] == categrade and class_id<info["classnum"]-1:
                self.class_info.append({
                    "categrade" : categrade,
                    "id" : class_id,
                    "name" : class_name
                })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)