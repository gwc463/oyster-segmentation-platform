package com.xai.oyster.controller;

import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/predict")
public class PredictController {
    @PostMapping
    public Map<String, Object> predict(@RequestParam("file") MultipartFile file, @RequestParam String model) {
        Map<String, Object> result = new HashMap<>();
        result.put("gender", "雌性");
        result.put("grayValue", 120);
        result.put("mask", "模拟掩膜数据");
        return result;
    }
}