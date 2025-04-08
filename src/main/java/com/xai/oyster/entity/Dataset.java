package com.xai.oyster.entity;

import lombok.Data;
import javax.persistence.*;
import java.time.LocalDateTime;

@Data
@Entity
@Table(name = "datasets")
public class Dataset {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    private String name;
    private String type;
    private int size;
    @ManyToOne
    @JoinColumn(name = "project_id")
    private Project project;
    private LocalDateTime createdAt;
}