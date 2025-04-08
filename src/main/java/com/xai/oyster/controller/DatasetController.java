package com.xai.oyster.controller;

import com.xai.oyster.entity.Dataset;
import com.xai.oyster.service.DatasetService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/datasets")
public class DatasetController {
    @Autowired
    private DatasetService datasetService;

    @PostMapping
    public Dataset createDataset(@RequestBody Dataset dataset) {
        return datasetService.createDataset(dataset);
    }

    @GetMapping
    public List<Dataset> getAllDatasets() {
        return datasetService.getAllDatasets();
    }

    @DeleteMapping("/{id}")
    public void deleteDataset(@PathVariable Long id) {
        datasetService.deleteDataset(id);
    }
}