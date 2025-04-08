package com.xai.oyster.service;

import com.xai.oyster.entity.Project;
import com.xai.oyster.repository.ProjectRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class ProjectService {
    @Autowired
    private ProjectRepository projectRepository;

    public Project createProject(Project project) {
        return projectRepository.save(project);
    }

    public List<Project> getAllProjects() {
        return projectRepository.findAll();
    }

    public void deleteProject(Long id) {
        projectRepository.deleteById(id);
    }
}